"""Cross-PLM scoring (score_model) orchestrator path.

Upstream PLMAlign topology:
- Parallel retrieval via Ankh-CL VDB + ESM-1b VDB.
- Union hit ids.
- Re-embed query + targets with ProtT5 for the score matrix.
- One alignment pass → one A3M.

This test pins that flow end-to-end against an httpx MockTransport that
inspects which model each /embed and /search call used, so regressions
where (say) targets accidentally get re-embedded with ankh_cl show up
immediately.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from plmmsa.pipeline.fetcher import DictTargetFetcher
from plmmsa.pipeline.orchestrator import Orchestrator, OrchestratorConfig


def _handler(calls: list[dict[str, Any]], shard_ids: set[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        calls.append({"path": path, "body": body})

        if path == "/embed":
            # Produce len(seq) vectors per sequence; the actual values
            # don't matter for topology verification, only that the model
            # name is preserved.
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "dim": 2,
                    "embeddings": [
                        [[1.0, 0.0] for _ in range(len(s))]
                        for s in body["sequences"]
                    ],
                },
            )
        if path == "/embed_by_id/bin":
            import numpy as np

            from plmmsa.align import binary as _binary

            found_vec = np.asarray([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
            found_ids = [rid for rid in body["ids"] if rid in shard_ids]
            missing = [rid for rid in body["ids"] if rid not in shard_ids]
            metadata = {
                "model": body["model"],
                "dim": 2,
                "found_ids": found_ids,
                "missing": missing,
            }
            frame = _binary.encode_tensors(metadata, [found_vec] * len(found_ids))
            return httpx.Response(
                200,
                content=frame,
                headers={"Content-Type": _binary.CONTENT_TYPE_EMBED},
            )
        if path == "/search":
            coll = body["collection"]
            # Each retrieval model's VDB returns a different, slightly
            # overlapping set of neighbors.
            if coll.startswith("ankh_cl"):
                results = [
                    {"id": "T_shared", "distance": 0.05},
                    {"id": "T_ankh_only", "distance": 0.1},
                ]
            elif coll.startswith("esm1b"):
                results = [
                    {"id": "T_shared", "distance": 0.04},
                    {"id": "T_esm_only", "distance": 0.2},
                ]
            else:
                results = []
            return httpx.Response(
                200,
                json={"collection": coll, "k": body["k"], "results": [results]},
            )
        if path == "/align":
            n = len(body["target_embeddings"])
            return httpx.Response(
                200,
                json={
                    "aligner": body["aligner"],
                    "mode": body["mode"],
                    "alignments": [
                        {
                            "score": 1.0 + i,
                            "mode": "local",
                            "query_start": 0,
                            "query_end": 1,
                            "target_start": 0,
                            "target_end": 1,
                            "columns": [[0, 0]],
                        }
                        for i in range(n)
                    ],
                },
            )
        return httpx.Response(404, json={"code": "unexpected", "message": path})

    return handler


def _orch(
    score_model: str,
    shard_models: frozenset[str],
    calls: list[dict[str, Any]],
    shard_ids: set[str] | None = None,
) -> Orchestrator:
    shard_ids = shard_ids or set()
    transport = httpx.MockTransport(_handler(calls, shard_ids))
    # Distinct target sequences so the test can distinguish target-embed
    # calls from query-embed calls (query sequence is "MK" in callers).
    fetcher = DictTargetFetcher({
        "T_shared": "MKT",
        "T_ankh_only": "MKTV",
        "T_esm_only": "MKTVL",
    })
    return Orchestrator(
        OrchestratorConfig(
            embedding_url="http://embedding",
            vdb_url="http://vdb",
            align_url="http://align",
            align_transport="json",
            score_model=score_model,
            shard_models=shard_models,
        ),
        fetcher,
        client_factory=lambda: httpx.AsyncClient(
            transport=transport, base_url="http://stub"
        ),
    )


async def test_cross_plm_runs_parallel_retrieval_then_single_align() -> None:
    """With `score_model=prott5` and retrieval via ankh_cl + esm1b:

    - Two parallel /search calls (ankh_uniref50 + esm1b_uniref50).
    - Two /embed calls for query (one per retrieval model).
    - One /embed call for the query with prott5.
    - One /embed call for targets with prott5.
    - One /align call.
    Hits are unioned into 3 unique ids (T_shared dedups).
    """
    calls: list[dict[str, Any]] = []
    orch = _orch("prott5", shard_models=frozenset(), calls=calls)
    result = await orch.run(
        {"sequences": ["MK"], "models": ["ankh_cl", "esm1b"], "k": 3}
    )

    # Count + shape of calls per kind.
    search_calls = [c for c in calls if c["path"] == "/search"]
    align_calls = [c for c in calls if c["path"] == "/align"]
    embed_calls = [c for c in calls if c["path"] == "/embed"]

    assert len(search_calls) == 2  # one per VDB
    assert len(align_calls) == 1  # one unified scoring pass
    # Embeds: 2 retrieval queries (ankh_cl + esm1b) + 1 prott5 query + 1
    # prott5 targets = 4. (Target chunking doesn't split at this size.)
    assert len(embed_calls) == 4

    # Query embeds go through the retrieval models.
    retrieval_query_embeds = [
        c for c in embed_calls if c["body"]["sequences"] == ["MK"]
    ]
    retrieval_models_seen = {c["body"]["model"] for c in retrieval_query_embeds}
    assert {"ankh_cl", "esm1b", "prott5"} == retrieval_models_seen

    # Target embed is ProtT5.
    target_embed = [c for c in embed_calls if c["body"]["sequences"] != ["MK"]]
    assert len(target_embed) == 1
    assert target_embed[0]["body"]["model"] == "prott5"

    # Result shape: union of hits is 3, alignment got 3 targets.
    assert result.stats["topology"] == "cross_plm"
    assert result.stats["hits_fetched"] == 3
    assert result.stats["retrieval_models"] == ["ankh_cl", "esm1b"]
    assert result.stats["score_model"] == "prott5"
    assert set(result.stats["per_retrieval"].keys()) == {"ankh_cl", "esm1b"}


async def test_cross_plm_uses_shard_store_when_configured() -> None:
    """With `shard_models={'prott5'}` and two of three ids in the store,
    target embedding makes one /embed_by_id call + one /embed (for the miss)."""
    calls: list[dict[str, Any]] = []
    orch = _orch(
        "prott5",
        shard_models=frozenset({"prott5"}),
        calls=calls,
        shard_ids={"T_shared", "T_ankh_only"},
    )
    await orch.run(
        {"sequences": ["MK"], "models": ["ankh_cl", "esm1b"], "k": 3}
    )

    paths = [c["path"] for c in calls]
    assert "/embed_by_id/bin" in paths

    # The miss-fallback /embed for targets should carry exactly one seq
    # (T_esm_only's sequence).
    target_embed_fallback = [
        c for c in calls
        if c["path"] == "/embed"
        and c["body"]["model"] == "prott5"
        and c["body"]["sequences"] != ["MK"]
    ]
    assert len(target_embed_fallback) == 1
    assert len(target_embed_fallback[0]["body"]["sequences"]) == 1


async def test_cross_plm_tolerates_one_vdb_failure() -> None:
    """One retrieval model's /embed or /search failing shouldn't nuke the
    whole run — the union proceeds from whatever returned."""
    # Wire a transport where the ankh_cl search URL 500s.
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append({"path": path, "body": body})

        if path == "/search" and body["collection"].startswith("ankh_cl"):
            return httpx.Response(500, json={"message": "synthetic"})
        # Delegate everything else to a working handler so the happy path
        # through esm1b + prott5 still produces an alignment.
        return _handler([], set())(request)

    transport = httpx.MockTransport(handler)
    orch = Orchestrator(
        OrchestratorConfig(
            embedding_url="http://embedding",
            vdb_url="http://vdb",
            align_url="http://align",
            align_transport="json",
            score_model="prott5",
        ),
        DictTargetFetcher({"T_shared": "MK", "T_esm_only": "MK"}),
        client_factory=lambda: httpx.AsyncClient(
            transport=transport, base_url="http://stub"
        ),
    )
    result = await orch.run(
        {"sequences": ["MK"], "models": ["ankh_cl", "esm1b"], "k": 3}
    )
    # Still succeeds — esm1b hits came through.
    assert result.stats["topology"] == "cross_plm"
    assert result.stats["per_retrieval"]["ankh_cl"].get("error") is not None
    assert result.stats["per_retrieval"]["esm1b"].get("error") is None


async def test_score_model_empty_falls_back_to_per_model_path() -> None:
    """With `score_model=""`, the old per-model-aggregate path runs.
    No cross-PLM topology, one alignment per retrieval model."""
    calls: list[dict[str, Any]] = []
    orch = _orch("", shard_models=frozenset(), calls=calls)
    result = await orch.run(
        {"sequences": ["MK"], "models": ["ankh_cl", "esm1b"], "k": 3}
    )
    assert result.stats.get("topology") != "cross_plm"
    # Two align calls (one per retrieval model).
    assert sum(1 for c in calls if c["path"] == "/align") == 2
