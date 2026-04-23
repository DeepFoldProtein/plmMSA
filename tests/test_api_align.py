from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _StubResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


class _StubAsyncClient:
    _response: _StubResponse | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> _StubAsyncClient:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(
        self,
        url: str,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> _StubResponse:
        assert url.endswith("/align"), f"unexpected upstream url: {url}"
        assert json is not None
        assert {"aligner", "mode", "query_embedding", "target_embeddings"} <= json.keys()
        assert _StubAsyncClient._response is not None
        return _StubAsyncClient._response


def test_v2_align_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align",
            json={
                "aligner": "plmalign",
                "mode": "local",
                "query_embedding": [[1.0, 0.0]],
                "target_embeddings": [[[1.0, 0.0]]],
            },
        )
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_v2_align_forwards_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setenv("ALIGN_URL", "http://stub-align:9999")

    _StubAsyncClient._response = _StubResponse(
        status_code=200,
        body={
            "aligner": "plmalign",
            "mode": "local",
            "alignments": [
                {
                    "score": 1.0,
                    "mode": "local",
                    "query_start": 0,
                    "query_end": 1,
                    "target_start": 0,
                    "target_end": 1,
                    "columns": [[0, 0]],
                }
            ],
        },
    )

    import plmmsa.api.routes.v2 as v2_mod

    monkeypatch.setattr(v2_mod.httpx, "AsyncClient", _StubAsyncClient)

    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/align",
            json={
                "aligner": "plmalign",
                "mode": "local",
                "query_embedding": [[1.0, 0.0]],
                "target_embeddings": [[[1.0, 0.0]]],
            },
            headers={"Authorization": "Bearer secret"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["aligner"] == "plmalign"
    assert body["alignments"][0]["score"] == 1.0
