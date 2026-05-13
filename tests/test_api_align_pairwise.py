"""Integration tests for `POST /v2/align/pairwise`.

Pattern matches `tests/test_api_templates_realign.py`: TestClient
against the live FastAPI app, with the orchestrator stubbed at the
module-level `_pairwise_orchestrator` handle. Unlike the rest of `/v2/`,
this endpoint is **public** -- no bearer token required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient


@dataclass(slots=True)
class _StubOrchestrator:
    """Minimal stub satisfying `PairwiseAlignOrchestrator.run`'s shape.

    Captures the last request for inspection. Returns canned alignments
    + stats from `result_alignments` / `result_stats` (set per test).
    """

    last_request: Any = None
    result_alignments: list[Any] = field(default_factory=list)
    result_stats: dict[str, Any] | None = None
    result_a3m: str | None = ""
    raise_with: Exception | None = None

    async def run(self, request: Any) -> Any:
        self.last_request = request
        if self.raise_with is not None:
            raise self.raise_with
        from plmmsa.align.pairwise import PairwiseAlignment, PairwiseAlignResult

        alignments: list[PairwiseAlignment] = self.result_alignments or [
            PairwiseAlignment(
                target_id="t1",
                score=0.5,
                columns=[(0, 0), (1, 1), (2, 2)],
                query_start=0,
                query_end=3,
                target_start=0,
                target_end=3,
                a3m_row="ABC",
            )
        ]
        stats = self.result_stats or {
            "pipeline": "align_pairwise",
            "query_id": "q",
            "query_length": 3,
            "targets_in": 1,
            "targets_kept": 1,
            "targets_dropped_no_match": 0,
            "unique_target_seqs": 1,
            "model": "ankh_large",
            "mode": "glocal",
            "aligner": "otalign",
            "sort_by_score": False,
            "emit_a3m": True,
        }
        return PairwiseAlignResult(
            alignments=alignments,
            stats=stats,
            a3m=self.result_a3m,
        )


@pytest.fixture
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch):
    import plmmsa.api.routes.v2 as v2

    stub = _StubOrchestrator()
    monkeypatch.setattr(v2, "_pairwise_orchestrator", stub)
    return stub


def test_no_token_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    """`/v2/align/pairwise` is the public surface -- requests without an
    Authorization header must succeed, not 401.
    """
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            json={
                "query_sequence": "ABC",
                "targets": [{"id": "t", "sequence": "ABC"}],
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stats"]["pipeline"] == "align_pairwise"
    assert body["alignments"][0]["target_id"] == "t1"


def test_happy_path_returns_alignments_and_stats(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    stub_orchestrator.result_a3m = ">t1 score:0.500\nABC\n"
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_id": "q",
                "query_sequence": "ABC",
                "targets": [{"id": "t1", "sequence": "ABC"}],
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["alignments"]) == 1
    a = body["alignments"][0]
    assert a["target_id"] == "t1"
    assert a["score"] == 0.5
    assert a["columns"] == [[0, 0], [1, 1], [2, 2]]
    assert a["query_start"] == 0 and a["query_end"] == 3
    assert a["a3m_row"] == "ABC"
    assert body["a3m"] == ">t1 score:0.500\nABC\n"
    assert body["stats"]["pipeline"] == "align_pairwise"
    assert body["stats"]["targets_kept"] == 1

    # Request reached the orchestrator unchanged.
    req = stub_orchestrator.last_request
    assert req.query_id == "q"
    assert req.query_sequence == "ABC"
    assert len(req.targets) == 1
    assert req.targets[0].target_id == "t1"
    assert req.targets[0].sequence == "ABC"
    assert req.model is None and req.mode is None and req.aligner is None
    assert req.emit_a3m is True
    assert req.sort_by_score is False


def test_request_overrides_pass_through(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_sequence": "ABC",
                "targets": [{"id": "t", "sequence": "ABC"}],
                "model": "ankh_cl",
                "mode": "q2t",
                "aligner": "otalign",
                "options": {"eps": 0.05},
                "emit_a3m": False,
                "sort_by_score": True,
            },
        )

    assert resp.status_code == 200
    req = stub_orchestrator.last_request
    assert req.model == "ankh_cl"
    assert req.mode == "q2t"
    assert req.aligner == "otalign"
    assert req.options == {"eps": 0.05}
    assert req.emit_a3m is False
    assert req.sort_by_score is True


def test_invalid_fasta_propagates_as_400(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.errors import ErrorCode, PlmMSAError

    stub_orchestrator.raise_with = PlmMSAError(
        "duplicate target_id 't'.",
        code=ErrorCode.INVALID_FASTA,
        http_status=400,
        detail={"target_id": "t"},
    )
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_sequence": "ABC",
                "targets": [
                    {"id": "t", "sequence": "ABC"},
                    {"id": "t", "sequence": "DEF"},
                ],
            },
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_INVALID_FASTA"
    assert (body.get("detail") or {}).get("target_id") == "t"


def test_seq_too_long_propagates_as_400(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.errors import ErrorCode, PlmMSAError

    stub_orchestrator.raise_with = PlmMSAError(
        "query too long",
        code=ErrorCode.SEQ_TOO_LONG,
        http_status=400,
    )
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_sequence": "A" * 2000,
                "targets": [{"id": "t", "sequence": "A"}],
            },
        )

    assert resp.status_code == 400
    assert resp.json()["code"] == "E_SEQ_TOO_LONG"


def test_missing_required_fields_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic validation rejects requests without `query_sequence` or
    `targets` before the orchestrator runs."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        # Missing `targets`.
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={"query_sequence": "ABC"},
        )

    assert resp.status_code == 422


def test_empty_targets_array_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`targets` has `min_length=1` at the Pydantic edge."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align/pairwise",
            headers={"Authorization": "Bearer secret"},
            json={"query_sequence": "ABC", "targets": []},
        )

    assert resp.status_code == 422
