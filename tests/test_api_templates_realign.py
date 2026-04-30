"""Integration tests for `POST /v2/templates/realign`.

Pattern matches `tests/test_api_align.py`: TestClient against the live
FastAPI app, with the orchestrator stubbed at the module-level
`_templates_orchestrator` handle. Auth is gated by `ADMIN_TOKEN`,
matching `/v2/embed` and `/v2/align`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient


@dataclass(slots=True)
class _StubOrchestrator:
    """Minimal stub satisfying `TemplatesRealignOrchestrator.run`'s shape.

    Captures the last request for inspection. Returns canned payload
    + stats from `result_payload` / `result_stats` (set per test).
    """

    last_request: Any = None
    result_payload: str = ">Q\nABC\n>t/1-3 Score=0.500\nABC\n"
    result_stats: dict[str, Any] = None  # type: ignore[assignment]
    raise_with: Exception | None = None

    def __post_init__(self) -> None:
        if self.result_stats is None:
            self.result_stats = {
                "pipeline": "templates_realign",
                "query_length": 3,
                "records_in": 1,
                "records_kept": 1,
                "records_dropped_sanity": 0,
                "records_dropped_no_match": 0,
                "unique_template_seqs": 1,
                "model": "ankh_large",
                "mode": "glocal",
                "aligner": "otalign",
            }

    async def run(self, request: Any) -> Any:
        self.last_request = request
        if self.raise_with is not None:
            raise self.raise_with
        from plmmsa.templates import TemplatesRealignResult

        return TemplatesRealignResult(
            payload=self.result_payload,
            stats=self.result_stats,
        )


@pytest.fixture
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch):
    """Swap the module-level `_templates_orchestrator` for a stub. The
    `_get_templates_orchestrator` helper just returns the cached value
    when set, so this short-circuits the production HttpTransport
    construction."""
    import plmmsa.api.routes.v2 as v2

    stub = _StubOrchestrator()
    monkeypatch.setattr(v2, "_templates_orchestrator", stub)
    return stub


def test_missing_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/templates/realign",
            json={
                "query_sequence": "ABC",
                "a3m": ">t/1-3\nABC\n",
            },
        )
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_happy_path_returns_payload_and_stats(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/templates/realign",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_id": "Q",
                "query_sequence": "ABC",
                "a3m": ">t/1-3\nABC\n",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["format"] == "a3m"
    assert body["payload"] == ">Q\nABC\n>t/1-3 Score=0.500\nABC\n"
    assert body["stats"]["records_kept"] == 1
    assert body["stats"]["mode"] == "glocal"

    # The orchestrator received the request body unchanged.
    req = stub_orchestrator.last_request
    assert req.query_id == "Q"
    assert req.query_sequence == "ABC"
    assert req.model is None
    assert req.mode is None


def test_request_overrides_pass_through(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    """Per-request `model` / `mode` / `options` arrive at the
    orchestrator without modification."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/templates/realign",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_id": "Q",
                "query_sequence": "ABC",
                "a3m": ">t/1-3\nABC\n",
                "model": "ankh_cl",
                "mode": "q2t",
                "options": {"eps": 0.05},
            },
        )

    assert resp.status_code == 200
    req = stub_orchestrator.last_request
    assert req.model == "ankh_cl"
    assert req.mode == "q2t"
    assert req.options == {"eps": 0.05}


def test_query_too_long_propagates_as_400(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    """Orchestrator-side `PlmMSAError(SEQ_TOO_LONG)` surfaces as the
    documented 400 with the stable code (the existing app-level
    error handler converts PlmMSAError → JSON)."""
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
            "/v2/templates/realign",
            headers={"Authorization": "Bearer secret"},
            json={
                "query_sequence": "ABCDE" * 300,  # length doesn't matter; stub raises
                "a3m": ">t/1-1\nA\n",
            },
        )

    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_SEQ_TOO_LONG"


def test_invalid_fasta_propagates_as_400(
    monkeypatch: pytest.MonkeyPatch,
    stub_orchestrator: _StubOrchestrator,
) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.errors import ErrorCode, PlmMSAError

    stub_orchestrator.raise_with = PlmMSAError(
        "query/a3m length mismatch",
        code=ErrorCode.INVALID_FASTA,
        http_status=400,
    )
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/templates/realign",
            headers={"Authorization": "Bearer secret"},
            json={"query_sequence": "ABC", "a3m": ">t/1-5\nABCDE\n"},
        )

    assert resp.status_code == 400
    assert resp.json()["code"] == "E_INVALID_FASTA"


def test_missing_required_fields_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic validation rejects requests without `query_sequence`
    or `a3m` before the orchestrator is touched."""
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/templates/realign",
            headers={"Authorization": "Bearer secret"},
            json={"query_sequence": "ABC"},  # missing `a3m`
        )

    assert resp.status_code == 422
