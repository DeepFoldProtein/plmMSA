from __future__ import annotations

from typing import Any

import httpx
import pytest

from plmmsa.pipeline.fetcher import DictTargetFetcher
from plmmsa.pipeline.orchestrator import Orchestrator, OrchestratorConfig


def _make_mock_transport(
    *,
    query_emb: list[list[float]],
    neighbors: list[dict[str, Any]],
    target_embs: list[list[list[float]]],
    alignments: list[dict[str, Any]],
) -> httpx.MockTransport:
    calls: list[tuple[str, dict[str, Any]]] = []
    target_state = {"embed_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = {} if not request.content else _json_loads(request.content)
        calls.append((path, body))

        if path == "/embed":
            target_state["embed_calls"] += 1
            if target_state["embed_calls"] == 1:
                return httpx.Response(
                    200, json={"model": body["model"], "dim": 2, "embeddings": [query_emb]}
                )
            return httpx.Response(
                200, json={"model": body["model"], "dim": 2, "embeddings": target_embs}
            )
        if path == "/search":
            return httpx.Response(
                200,
                json={"collection": body["collection"], "k": body["k"], "results": [neighbors]},
            )
        if path == "/align":
            return httpx.Response(
                200,
                json={"aligner": body["aligner"], "mode": body["mode"], "alignments": alignments},
            )
        return httpx.Response(404, json={"code": "unexpected", "message": path})

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def _json_loads(raw: bytes) -> dict[str, Any]:
    import json

    return json.loads(raw.decode("utf-8"))


def _make_orchestrator(transport: httpx.MockTransport, fetcher) -> Orchestrator:
    config = OrchestratorConfig(
        embedding_url="http://embedding",
        vdb_url="http://vdb",
        align_url="http://align",
        align_transport="json",
    )

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url="http://stub")

    return Orchestrator(config, fetcher, client_factory=client_factory)


async def test_orchestrator_happy_path_builds_a3m() -> None:
    query_seq = "MKT"
    target_seq = "MKTA"  # one extra residue → lowercase insertion

    transport = _make_mock_transport(
        query_emb=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        neighbors=[{"id": "T1", "distance": 0.1}],
        target_embs=[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]],
        alignments=[
            {
                "score": 2.5,
                "mode": "local",
                "query_start": 0,
                "query_end": 3,
                "target_start": 0,
                "target_end": 4,
                "columns": [[0, 0], [1, 1], [2, 2], [-1, 3]],
            }
        ],
    )
    fetcher = DictTargetFetcher({"T1": target_seq})
    orchestrator = _make_orchestrator(transport, fetcher)

    result = await orchestrator.run({"sequences": [query_seq], "model": "ankh_cl"})

    assert result.format == "a3m"
    lines = result.payload.rstrip("\n").splitlines()
    # Default query_id is now "101" (numeric label, chain index + 101).
    # Previously defaulted to "query"; orchestrator still accepts an
    # explicit `query_id=` request field.
    assert lines[0].startswith(">101")
    assert lines[1] == "MKT"
    assert lines[2] == ">T1   2.500"
    assert lines[3] == "MKTa"
    assert result.stats["hits_fetched"] == 1
    assert result.stats["hits_found"] == 1
    assert result.stats["pipeline"] == "orchestrator"


async def test_orchestrator_falls_back_to_query_only_when_no_targets() -> None:
    transport = _make_mock_transport(
        query_emb=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        neighbors=[{"id": "UNKNOWN", "distance": 0.1}],
        target_embs=[],
        alignments=[],
    )
    fetcher = DictTargetFetcher({})
    orchestrator = _make_orchestrator(transport, fetcher)

    result = await orchestrator.run({"sequences": ["MKT"]})

    assert result.format == "a3m"
    assert result.payload == ">101   3.000\nMKT\n"
    assert result.stats["hits_found"] == 1
    assert result.stats["hits_fetched"] == 0
    assert "note" in result.stats


async def test_orchestrator_rejects_empty_sequences() -> None:
    orchestrator = _make_orchestrator(
        _make_mock_transport(query_emb=[], neighbors=[], target_embs=[], alignments=[]),
        DictTargetFetcher({}),
    )
    with pytest.raises(ValueError, match="empty"):
        await orchestrator.run({"sequences": []})
