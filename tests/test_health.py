from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_health_cache() -> None:
    """Clear the module-level TTL cache between tests so one failure mode
    doesn't leak into the next test's response."""
    from plmmsa.api import health as _health

    _health._aggregate_cache["ts"] = 0.0
    _health._aggregate_cache["payload"] = None


def test_healthz_ok() -> None:
    """Bare liveness probe — returns 200 regardless of downstream state.
    This is what docker-compose + CF edge probes hit."""
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "api"


def _stub_probes(monkeypatch: pytest.MonkeyPatch, *, http_ok: bool, redis_ok: bool) -> None:
    """Replace the downstream probes with deterministic stubs."""
    from plmmsa.api import health as _health

    async def _fake_http(client: Any, service: str, url: str) -> tuple[str, Any]:
        return service, _health.DownstreamStatus(
            status="ok" if http_ok else "down",
            detail={"stub": True} if http_ok else {"error": "stubbed"},
        )

    async def _fake_redis(service: str, url: str) -> tuple[str, Any]:
        return service, _health.DownstreamStatus(
            status="ok" if redis_ok else "down",
            detail=None if redis_ok else {"error": "stubbed"},
        )

    monkeypatch.setattr(_health, "_probe_http", _fake_http)
    monkeypatch.setattr(_health, "_probe_redis", _fake_redis)


def test_health_aggregate_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every downstream reports ok, the aggregate is ok with a
    per-service status map."""
    _stub_probes(monkeypatch, http_ok=True, redis_ok=True)
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "api"
    assert set(body["downstream"].keys()) == {
        "embedding",
        "vdb",
        "align",
        "cache-ops",
        "cache-seq",
        "cache-emb",
    }
    for name, entry in body["downstream"].items():
        assert entry["status"] == "ok", f"{name} should be ok"


def test_health_aggregate_downstream_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any downstream `down` demotes the aggregate to `down` — a warmup
    gate must not accept work when any dependency is unreachable."""
    _stub_probes(monkeypatch, http_ok=False, redis_ok=True)
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "down"
    assert body["downstream"]["embedding"]["status"] == "down"
    assert body["downstream"]["cache-ops"]["status"] == "ok"


def test_health_aggregate_caches_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Burst polls within the TTL window reuse the cached payload and
    don't re-probe downstreams."""
    from plmmsa.api import app
    from plmmsa.api import health as _health

    call_counts: dict[str, int] = {"http": 0, "redis": 0}

    async def _counting_http(client: Any, service: str, url: str) -> tuple[str, Any]:
        call_counts["http"] += 1
        return service, _health.DownstreamStatus(status="ok")

    async def _counting_redis(service: str, url: str) -> tuple[str, Any]:
        call_counts["redis"] += 1
        return service, _health.DownstreamStatus(status="ok")

    monkeypatch.setattr(_health, "_probe_http", _counting_http)
    monkeypatch.setattr(_health, "_probe_redis", _counting_redis)

    with TestClient(app) as client:
        for _ in range(4):
            resp = client.get("/health")
            assert resp.status_code == 200
    # First request probes 3 HTTP + 3 Redis; next three should hit cache.
    assert call_counts == {"http": 3, "redis": 3}


def test_v1_gone() -> None:
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/v1/anything")
    assert resp.status_code == 410
    body = resp.json()
    assert body["code"] == "E_GONE"
    assert body["detail"]["successor"] == "/v2/"


def test_v2_version() -> None:
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/v2/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api"] == "v2"
    assert "ankh_cl" in body["models"]


# /v2/msa lifecycle is tested end-to-end in test_api_msa.py with a
# monkeypatched fake-redis JobStore.
