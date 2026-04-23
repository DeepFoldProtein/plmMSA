"""Multi-model orchestration: two PLMs run in parallel, hits union by id.

Uses an httpx MockTransport that routes per-model traffic by inspecting the
`model` field in each /embed / /search / /align payload, so the test doesn't
depend on call ordering.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from plmmsa.pipeline.fetcher import DictTargetFetcher
from plmmsa.pipeline.orchestrator import Orchestrator, OrchestratorConfig


def _make_router(
    *,
    per_model: dict[str, dict[str, Any]],
) -> httpx.MockTransport:
    """Build a transport whose behavior branches on (path, model/collection).

    `per_model[model]` keys:
      - `query_emb` / `target_embs` (both used on /embed)
      - `neighbors` (used on /search)
      - `alignments` (used on /align)
      - `collection` (used to match /search payloads by collection)
    """

    embed_state = {m: 0 for m in per_model}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        if path == "/embed":
            m = body["model"]
            embed_state[m] += 1
            spec = per_model[m]
            if embed_state[m] == 1:
                return httpx.Response(
                    200,
                    json={"model": m, "dim": 2, "embeddings": [spec["query_emb"]]},
                )
            return httpx.Response(
                200,
                json={"model": m, "dim": 2, "embeddings": spec["target_embs"]},
            )
        if path == "/search":
            # Route by collection → model (the orchestrator uses
            # `<model>_uniref50` by default).
            coll = body["collection"]
            match = next(
                (m for m, spec in per_model.items() if spec["collection"] == coll),
                None,
            )
            assert match is not None, f"unexpected collection {coll}"
            return httpx.Response(
                200,
                json={
                    "collection": coll,
                    "k": body["k"],
                    "results": [per_model[match]["neighbors"]],
                },
            )
        if path == "/align":
            # The align service doesn't know which model produced the
            # embeddings; target_embeddings[0] is the query self-score
            # request, so use the first real target as the sentinel.
            sentinel = body["target_embeddings"][1][0][0]
            match = next(
                (m for m, spec in per_model.items() if spec["sentinel"] == sentinel),
                None,
            )
            assert match is not None, f"unknown sentinel {sentinel}"
            self_alignment = {
                "score": 9.0,
                "mode": "local",
                "query_start": 0,
                "query_end": 1,
                "target_start": 0,
                "target_end": 1,
                "columns": [[0, 0]],
            }
            return httpx.Response(
                200,
                json={
                    "aligner": body["aligner"],
                    "mode": body["mode"],
                    "alignments": [self_alignment, *per_model[match]["alignments"]],
                },
            )
        return httpx.Response(404, json={"code": "unexpected", "message": path})

    return httpx.MockTransport(handler)


async def test_two_models_union_hits_keeping_best_score() -> None:
    # ankh_cl returns T1 + T2 (T1 score 5.0, T2 score 3.0)
    # esm1b   returns T1 + T3 (T1 score 7.0, T3 score 2.0)
    # Union should keep T1 from esm1b (7.0), T2 from ankh_cl, T3 from esm1b.
    per_model = {
        "ankh_cl": {
            "collection": "ankh_cl_uniref50",
            "sentinel": 11.0,
            "query_emb": [[1.0, 0.0], [0.0, 1.0]],
            "neighbors": [{"id": "T1", "distance": 0.1}, {"id": "T2", "distance": 0.2}],
            "target_embs": [[[11.0, 0.0]], [[11.0, 0.0]]],
            "alignments": [
                {
                    "score": 5.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                },
                {
                    "score": 3.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                },
            ],
        },
        "esm1b": {
            "collection": "esm1b_uniref50",
            "sentinel": 22.0,
            "query_emb": [[1.0, 0.0], [0.0, 1.0]],
            "neighbors": [{"id": "T1", "distance": 0.1}, {"id": "T3", "distance": 0.3}],
            "target_embs": [[[22.0, 0.0]], [[22.0, 0.0]]],
            "alignments": [
                {
                    "score": 7.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                },
                {
                    "score": 2.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                },
            ],
        },
    }

    transport = _make_router(per_model=per_model)
    fetcher = DictTargetFetcher({"T1": "MK", "T2": "MK", "T3": "MK"})
    orch = Orchestrator(
        OrchestratorConfig(
            embedding_url="http://embedding",
            vdb_url="http://vdb",
            align_url="http://align",
            align_transport="json",
        ),
        fetcher,
        client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://stub"),
    )

    result = await orch.run({"sequences": ["MK"], "models": ["ankh_cl", "esm1b"]})

    lines = result.payload.rstrip("\n").splitlines()
    # Query is line 0; remaining headers are hits ordered by score desc.
    # First header is the query record; the rest are hits in score order.
    headers = [line for line in lines if line.startswith(">")]
    hit_ids = [h.split()[0][1:] for h in headers[1:]]
    # Scores: T1=7.0 (esm1b beat ankh_cl's 5.0), T2=3.0 (ankh_cl), T3=2.0 (esm1b).
    assert hit_ids == ["T1", "T2", "T3"], f"unexpected hit order: {hit_ids}"

    # Stats shape: per-model breakdown + union numbers.
    assert result.stats["models"] == ["ankh_cl", "esm1b"]
    assert result.stats["hits_fetched"] == 3  # T1, T2, T3 after union
    assert result.stats["hits_fetched_before_union"] == 4  # 2 + 2
    assert "per_model" in result.stats
    assert result.stats["per_model"]["ankh_cl"]["hits_fetched"] == 2
    assert result.stats["per_model"]["esm1b"]["hits_fetched"] == 2

    # The best T1 score wins (7.0, from esm1b).
    t1_header = next(h for h in headers if h.split()[0] == ">T1")
    assert float(t1_header.split()[1]) == 7.0


async def test_single_model_legacy_path_still_works() -> None:
    """Passing `model="..."` (without `models`) keeps the old behavior:
    one pipeline run, no per-model stats key drift."""
    per_model = {
        "ankh_cl": {
            "collection": "ankh_cl_uniref50",
            "sentinel": 11.0,
            "query_emb": [[1.0, 0.0], [0.0, 1.0]],
            "neighbors": [{"id": "T1", "distance": 0.1}],
            "target_embs": [[[11.0, 0.0]]],
            "alignments": [
                {
                    "score": 5.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                },
            ],
        },
    }
    transport = _make_router(per_model=per_model)
    fetcher = DictTargetFetcher({"T1": "MK"})
    orch = Orchestrator(
        OrchestratorConfig(
            embedding_url="http://embedding",
            vdb_url="http://vdb",
            align_url="http://align",
            align_transport="json",
        ),
        fetcher,
        client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://stub"),
    )

    result = await orch.run({"sequences": ["MK"], "model": "ankh_cl"})

    assert result.stats["models"] == ["ankh_cl"]
    assert result.stats["hits_fetched"] == 1
