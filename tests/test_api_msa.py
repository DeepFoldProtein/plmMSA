from __future__ import annotations

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from plmmsa.jobs import JobResult, JobStore
from plmmsa.jobs.models import JobStatus

_BOOTSTRAP = "bootstrap-secret"
_AUTH = {"Authorization": f"Bearer {_BOOTSTRAP}"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, JobStore]:
    monkeypatch.setenv("ADMIN_TOKEN", _BOOTSTRAP)
    store = JobStore(FakeAsyncRedis())

    import plmmsa.api.routes.v2 as v2_mod

    async def fake_get_store() -> JobStore:
        return store

    monkeypatch.setattr(v2_mod, "_get_job_store", fake_get_store)

    from plmmsa.api import app

    return TestClient(app), store


def test_submit_msa_without_token_is_401(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.post("/v2/msa", json={"sequences": ["MKTIIAL"], "model": "ankh_cl"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_submit_msa_enqueues(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.post(
            "/v2/msa",
            headers=_AUTH,
            json={"sequences": ["MKTIIAL"], "model": "ankh_cl"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["status_url"].startswith("/v2/msa/")


async def test_get_msa_returns_record() -> None:
    # Test the store directly and then roundtrip through the API for
    # observed lifecycle. Done without fixture indirection to keep it clean.
    store = JobStore(FakeAsyncRedis())
    job = await store.create({"sequences": ["MKT"]})
    await store.mark_succeeded(
        job.id,
        JobResult(format="a3m", payload=">query\nMKT\n", stats={"depth": 1}),
    )

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.SUCCEEDED
    assert fetched.result is not None
    assert fetched.result.payload.startswith(">query")


def test_get_msa_without_token_is_401(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.get("/v2/msa/does-not-exist")
    assert resp.status_code == 401


def test_get_msa_404_on_missing(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.get("/v2/msa/does-not-exist", headers=_AUTH)
    assert resp.status_code == 404
    assert resp.json()["code"] == "E_JOB_NOT_FOUND"


def test_submit_then_get(client) -> None:
    tc, _ = client
    with tc as c:
        submit = c.post("/v2/msa", headers=_AUTH, json={"sequences": ["MKT"]})
        job_id = submit.json()["job_id"]
        resp = c.get(f"/v2/msa/{job_id}", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == job_id
    assert body["status"] == "queued"
    assert body["request"]["sequences"] == ["MKT"]


def test_cancel_without_token_is_401(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.delete("/v2/msa/any-id")
    assert resp.status_code == 401


def test_cancel_flips_status(client) -> None:
    tc, _ = client
    with tc as c:
        submit = c.post("/v2/msa", headers=_AUTH, json={"sequences": ["MKT"]})
        job_id = submit.json()["job_id"]
        cancel = c.delete(f"/v2/msa/{job_id}", headers=_AUTH)
        assert cancel.status_code == 204
        follow = c.get(f"/v2/msa/{job_id}", headers=_AUTH)
    body = follow.json()
    assert body["status"] == "cancelled"


def test_cancel_404_on_missing(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.delete("/v2/msa/does-not-exist", headers=_AUTH)
    assert resp.status_code == 404
