"""Orchestrator integration: shard-first target fetch.

Drives the full pipeline through an httpx MockTransport. One model (prott5)
is opted into `shard_models`; the /embed_by_id handler returns embeddings
for 2 of 3 target ids, and /embed must be called exactly once for the
third. Output order must align with `kept_ids` so the align step sees the
correct embedding in each position.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from plmmsa.pipeline.fetcher import DictTargetFetcher
from plmmsa.pipeline.orchestrator import Orchestrator, OrchestratorConfig


def _handler(calls: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        calls.append({"path": path, "body": body})

        if path == "/embed":
            # Query embedding (len rows per sequence) or miss fallback.
            # Orchestrator's _embed_query cross-checks that the returned
            # embedding length matches the query residue count, so the
            # stub must produce one vector per residue.
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "dim": 2,
                    "embeddings": [[[7.0, 7.0] for _ in range(len(s))] for s in body["sequences"]],
                },
            )
        if path == "/embed_by_id/bin":
            # Orchestrator now hits the binary endpoint. Build a frame
            # that matches the production encoder.
            import numpy as np

            from plmmsa.align import binary as _binary

            found_map = {
                "T1": np.asarray([[1.0, 0.0]], dtype=np.float32),
                "T3": np.asarray([[3.0, 0.0]], dtype=np.float32),
            }
            found_ids = [rid for rid in body["ids"] if rid in found_map]
            missing = [rid for rid in body["ids"] if rid not in found_map]
            metadata = {
                "model": body["model"],
                "dim": 2,
                "found_ids": found_ids,
                "missing": missing,
            }
            frame = _binary.encode_tensors(metadata, [found_map[rid] for rid in found_ids])
            return httpx.Response(
                200,
                content=frame,
                headers={"Content-Type": _binary.CONTENT_TYPE_EMBED},
            )
        if path == "/search":
            return httpx.Response(
                200,
                json={
                    "collection": body["collection"],
                    "k": body["k"],
                    "results": [
                        [
                            {"id": "T1", "distance": 0.1},
                            {"id": "T2", "distance": 0.2},
                            {"id": "T3", "distance": 0.3},
                        ]
                    ],
                },
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


def _orch(shard_models: frozenset[str], calls: list[dict[str, Any]]) -> Orchestrator:
    transport = httpx.MockTransport(_handler(calls))
    return Orchestrator(
        OrchestratorConfig(
            embedding_url="http://embedding",
            vdb_url="http://vdb",
            align_url="http://align",
            align_transport="json",
            shard_models=shard_models,
        ),
        DictTargetFetcher({"T1": "MK", "T2": "MK", "T3": "MK"}),
        client_factory=lambda: httpx.AsyncClient(transport=transport, base_url="http://stub"),
    )


async def test_shard_first_then_embed_for_misses() -> None:
    calls: list[dict[str, Any]] = []
    orch = _orch(frozenset({"prott5"}), calls)
    result = await orch.run({"sequences": ["MK"], "models": ["prott5"], "k": 3})

    # Expected call sequence:
    #   /embed (query),
    #   /search,
    #   /embed_by_id/bin (T1, T2, T3 — binary response),
    #   /embed (miss fallback: T2 only),
    #   /align (3 targets in kept_ids order)
    paths = [c["path"] for c in calls]
    assert paths == ["/embed", "/search", "/embed_by_id/bin", "/embed", "/align"]

    # The miss-fallback /embed call should carry only the miss sequence.
    miss_call = calls[3]
    assert miss_call["body"]["sequences"] == ["MK"]  # T2's sequence

    # Align should receive 3 target embeddings in kept_ids order (T1, T2, T3).
    align_call = calls[4]
    assert len(align_call["body"]["target_embeddings"]) == 3
    # T1 from shard = [[1.0, 0.0]]
    assert align_call["body"]["target_embeddings"][0][0][0] == 1.0
    # T2 from /embed fallback = [[7.0, 7.0]]
    assert align_call["body"]["target_embeddings"][1][0][0] == 7.0
    # T3 from shard = [[3.0, 0.0]]
    assert align_call["body"]["target_embeddings"][2][0][0] == 3.0

    # Final A3M should still have 3 hits.
    assert result.stats["hits_fetched"] == 3


async def test_no_shard_models_keeps_today_behavior() -> None:
    """With shard_models=frozenset(), the orchestrator never hits /embed_by_id."""
    calls: list[dict[str, Any]] = []
    orch = _orch(frozenset(), calls)
    await orch.run({"sequences": ["MK"], "models": ["prott5"], "k": 3})

    paths = [c["path"] for c in calls]
    assert "/embed_by_id/bin" not in paths
    # Expected: /embed (query), /search, /embed (targets), /align
    assert paths == ["/embed", "/search", "/embed", "/align"]
