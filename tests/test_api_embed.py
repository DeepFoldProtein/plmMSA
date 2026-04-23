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
    """Minimal httpx.AsyncClient replacement for the /v2/embed forward test."""

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
        assert url.endswith("/embed"), f"unexpected upstream url: {url}"
        assert json is not None
        assert "model" in json and "sequences" in json
        assert _StubAsyncClient._response is not None
        return _StubAsyncClient._response


def test_v2_embed_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post("/v2/embed", json={"model": "ankh_cl", "sequences": ["MKT"]})
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_v2_embed_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/embed",
            json={"model": "ankh_cl", "sequences": ["MKT"]},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_INVALID"


def test_v2_embed_forwards_upstream_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setenv("EMBEDDING_URL", "http://stub-embed:9999")

    _StubAsyncClient._response = _StubResponse(
        status_code=200,
        body={"model": "ankh_cl", "dim": 4, "embeddings": [[[0.0, 0.0, 0.0, 0.0]]]},
    )

    import plmmsa.api.routes.v2 as v2_mod

    monkeypatch.setattr(v2_mod.httpx, "AsyncClient", _StubAsyncClient)

    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/embed",
            json={"model": "ankh_cl", "sequences": ["A"]},
            headers={"Authorization": "Bearer secret"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "ankh_cl"
    assert body["dim"] == 4
    assert body["embeddings"] == [[[0.0, 0.0, 0.0, 0.0]]]
