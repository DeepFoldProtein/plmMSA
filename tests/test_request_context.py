"""Coverage for cross-service X-Request-ID propagation.

Three surfaces under test:

- **ContextVar + header helper** — the low-level primitive that keeps
  the current request id accessible from anywhere in a task.
- **Sidecar middleware** — accepts an incoming X-Request-ID or mints
  a fresh one, echoes it on the response, binds the ContextVar.
- **api stamps request_id onto the job payload** — so the worker can
  later rebind it and forward on downstream calls. The canonical
  cache-key computation must *not* include request_id.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plmmsa.jobs import JobStore, cache_key
from plmmsa.request_context import (
    RequestContextMiddleware,
    bind_request_id,
    current_request_id,
    httpx_headers_with_request_id,
)

_BOOTSTRAP = "bootstrap-secret"


def test_bind_and_read_request_id_in_context() -> None:
    bind_request_id("rid-abc")
    assert current_request_id() == "rid-abc"
    bind_request_id(None)
    assert current_request_id() is None


def test_httpx_headers_injects_bound_request_id() -> None:
    bind_request_id("rid-123")
    try:
        out = httpx_headers_with_request_id()
        assert out == {"X-Request-ID": "rid-123"}

        # Existing caller headers are preserved + merged.
        merged = httpx_headers_with_request_id({"Content-Type": "application/json"})
        assert merged == {
            "Content-Type": "application/json",
            "X-Request-ID": "rid-123",
        }

        # Caller's explicit X-Request-ID wins.
        explicit = httpx_headers_with_request_id({"X-Request-ID": "caller"})
        assert explicit == {"X-Request-ID": "caller"}
    finally:
        bind_request_id(None)


def test_sidecar_middleware_mints_and_echoes_request_id() -> None:
    """When no caller supplies X-Request-ID, the middleware mints one,
    echoes it on the response, and exposes it via the ContextVar to
    handlers."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, service="fake")

    @app.get("/")
    async def _h() -> dict[str, Any]:
        return {"rid_in_handler": current_request_id()}

    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    echoed = resp.headers.get("X-Request-ID")
    assert echoed is not None and len(echoed) > 0
    assert resp.json()["rid_in_handler"] == echoed


def test_sidecar_middleware_adopts_supplied_request_id() -> None:
    """When the api forwards X-Request-ID, the sidecar must use that
    exact value so the api log and sidecar log correlate."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, service="fake")

    @app.get("/")
    async def _h() -> dict[str, Any]:
        return {"rid": current_request_id()}

    with TestClient(app) as client:
        resp = client.get("/", headers={"X-Request-ID": "from-api-edge"})
    assert resp.headers["X-Request-ID"] == "from-api-edge"
    assert resp.json()["rid"] == "from-api-edge"


def test_cache_key_excludes_request_id() -> None:
    """Two submits with identical scientific parameters but different
    request ids must land on the same cache key — otherwise every retry
    would blow the cache."""
    with_rid = {"sequences": ["MKT"], "models": ["ankh_cl"], "request_id": "a"}
    without = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    other_rid = {"sequences": ["MKT"], "models": ["ankh_cl"], "request_id": "b"}
    assert cache_key(with_rid) == cache_key(without) == cache_key(other_rid)


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, JobStore]:
    monkeypatch.setenv("ADMIN_TOKEN", _BOOTSTRAP)
    store = JobStore(FakeAsyncRedis())

    import plmmsa.api.routes.v2 as v2_mod
    from plmmsa.jobs import ResultCache

    async def fake_get_store() -> JobStore:
        return store

    async def fake_get_cache() -> ResultCache:
        return ResultCache(None, ttl_s=0)  # disabled — isolate this test

    monkeypatch.setattr(v2_mod, "_get_job_store", fake_get_store)
    monkeypatch.setattr(v2_mod, "_get_result_cache", fake_get_cache)

    from plmmsa.api import app

    return TestClient(app), store


def test_api_submit_stamps_request_id_onto_job(api_client) -> None:
    """POST /v2/msa must persist the X-Request-ID under `job.request[
    "request_id"]` so the worker can rebind and forward it on every
    downstream call."""
    tc, store = api_client
    with tc as c:
        resp = c.post(
            "/v2/msa",
            headers={"X-Request-ID": "edge-trace-id"},
            json={"sequences": ["MKTIIAL"], "models": ["ankh_cl"]},
        )
    assert resp.status_code == 202, resp.json()
    assert resp.headers["X-Request-ID"] == "edge-trace-id"

    job_id = resp.json()["job_id"]
    job = asyncio.run(store.get(job_id))
    assert job is not None
    assert job.request.get("request_id") == "edge-trace-id"
