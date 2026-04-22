from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok() -> None:
    from plmmsa.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "api"


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
