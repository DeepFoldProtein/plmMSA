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
        assert url.endswith("/search"), f"unexpected upstream url: {url}"
        assert json is not None
        assert "collection" in json and "vectors" in json
        assert _StubAsyncClient._response is not None
        return _StubAsyncClient._response


def test_v2_search_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/search",
            json={"collection": "ankh_uniref50", "vectors": [[0.0, 0.0]], "k": 3},
        )
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_v2_search_forwards_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.setenv("VDB_URL", "http://stub-vdb:9999")

    _StubAsyncClient._response = _StubResponse(
        status_code=200,
        body={
            "collection": "ankh_uniref50",
            "k": 2,
            "results": [[{"id": "UR50_0000", "distance": 0.5}]],
        },
    )

    import plmmsa.api.routes.v2 as v2_mod

    monkeypatch.setattr(v2_mod.httpx, "AsyncClient", _StubAsyncClient)

    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/v2/search",
            json={"collection": "ankh_uniref50", "vectors": [[0.1, 0.2]], "k": 2},
            headers={"Authorization": "Bearer secret"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "ankh_uniref50"
    assert body["results"][0][0]["id"] == "UR50_0000"
