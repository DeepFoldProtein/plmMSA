from __future__ import annotations

import io
import tarfile

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from plmmsa.jobs import JobResult, JobStore


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, JobStore]:
    """ColabFold-compat routes reuse `_get_job_store` from the v2 module;
    swap in a FakeAsyncRedis-backed store so tests don't need a real Redis.
    """
    store = JobStore(FakeAsyncRedis())

    import plmmsa.api.routes.v2 as v2_mod

    async def fake_get_store() -> JobStore:
        return store

    monkeypatch.setattr(v2_mod, "_get_job_store", fake_get_store)

    from plmmsa.api import app

    return TestClient(app), store


def _submit_raw(tc: TestClient, prefix: str, q: str) -> dict:
    """POST /v2/colabfold/<prefix>/ticket/msa with a form body; return JSON."""
    with tc as c:
        resp = c.post(
            f"/v2/colabfold/{prefix}/ticket/msa",
            data={"q": q, "mode": "env-pairgreedy", "database": "uniref"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_plmmsa_ticket_msa_enqueues(client) -> None:
    tc, store = client
    body = _submit_raw(tc, "plmmsa", "MKTIIAL")
    assert body["status"] == "PENDING"
    job_id = body["id"]

    # The stamped aligner on the job record IS the flavor (PLMAlign).
    import asyncio

    job = asyncio.get_event_loop().run_until_complete(store.get(job_id))
    assert job is not None
    assert job.request["aligner"] == "plmalign"
    assert job.request["sequences"] == ["MKTIIAL"]
    # Tag-through so audit logs can see CF-origin jobs.
    assert job.request["_cf_flavor"] == "plmalign"


def test_otalign_ticket_msa_enqueues_with_otalign_flavor(client) -> None:
    tc, store = client
    body = _submit_raw(tc, "otalign", "MKTIIAL")
    job_id = body["id"]

    import asyncio

    job = asyncio.get_event_loop().run_until_complete(store.get(job_id))
    assert job is not None
    assert job.request["aligner"] == "otalign"
    assert job.request["_cf_flavor"] == "otalign"


def test_ticket_msa_accepts_fasta_body(client) -> None:
    tc, store = client
    fasta = ">T1104\nSGGGMTYHVLVQFDVPSDKRE\nAFAAAGL"
    body = _submit_raw(tc, "plmmsa", fasta)
    job_id = body["id"]

    import asyncio

    job = asyncio.get_event_loop().run_until_complete(store.get(job_id))
    assert job is not None
    assert job.request["sequences"] == ["SGGGMTYHVLVQFDVPSDKREAFAAAGL"]


def test_ticket_status_maps_running(client) -> None:
    tc, store = client
    body = _submit_raw(tc, "plmmsa", "MKTIIAL")
    job_id = body["id"]

    import asyncio

    asyncio.get_event_loop().run_until_complete(store.mark_running(job_id))

    with tc as c:
        resp = c.get(f"/v2/colabfold/plmmsa/ticket/msa/{job_id}")
    assert resp.status_code == 200
    assert resp.json() == {"id": job_id, "status": "RUNNING"}


def test_ticket_status_maps_succeeded_to_complete(client) -> None:
    tc, store = client
    body = _submit_raw(tc, "plmmsa", "MKTIIAL")
    job_id = body["id"]

    import asyncio

    asyncio.get_event_loop().run_until_complete(
        store.mark_succeeded(
            job_id,
            JobResult(format="a3m", payload=">q\nMKTIIAL\n", stats={"depth": 1}),
        )
    )

    with tc as c:
        resp = c.get(f"/v2/colabfold/plmmsa/ticket/msa/{job_id}")
    assert resp.json()["status"] == "COMPLETE"


def test_ticket_status_unknown_id_returns_unknown(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.get("/v2/colabfold/plmmsa/ticket/msa/nope-not-a-job")
    assert resp.status_code == 200
    # CF's contract: unknown ids resolve to status=UNKNOWN, not 404.
    assert resp.json() == {"id": "nope-not-a-job", "status": "UNKNOWN"}


def test_result_download_tar_layout(client) -> None:
    tc, store = client
    body = _submit_raw(tc, "plmmsa", "MKTIIAL")
    job_id = body["id"]

    import asyncio

    a3m = ">q\nMKTIIAL\n>hit1\nMKTIIA-\n"
    asyncio.get_event_loop().run_until_complete(
        store.mark_succeeded(
            job_id,
            JobResult(format="a3m", payload=a3m, stats={"depth": 2}),
        )
    )

    with tc as c:
        resp = c.get(f"/v2/colabfold/plmmsa/result/download/{job_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/octet-stream")

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r") as tf:
        names = sorted(m.name for m in tf.getmembers())
    assert names == sorted(
        [
            "uniref.a3m",
            "bfd.mgnify30.metaeuk30.smag30.a3m",
            "pair.a3m",
        ]
    )

    # uniref.a3m bytes equal the plmMSA payload.
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r") as tf:
        f = tf.extractfile("uniref.a3m")
        assert f is not None
        assert f.read().decode() == a3m


def test_result_download_missing_result_404s(client) -> None:
    tc, _store = client
    body = _submit_raw(tc, "plmmsa", "MKTIIAL")
    job_id = body["id"]
    # No mark_succeeded — result is None.
    with tc as c:
        resp = c.get(f"/v2/colabfold/plmmsa/result/download/{job_id}")
    assert resp.status_code == 404
    assert resp.json()["code"] == "E_JOB_NOT_FOUND"


def test_ticket_pair_returns_501(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.post(
            "/v2/colabfold/plmmsa/ticket/pair",
            data={"q": "MKT", "mode": "pair"},
        )
    assert resp.status_code == 501
    body = resp.json()
    assert body["status"] == "ERROR"
    assert "paired" in body["message"].lower()


def test_ticket_msa_missing_q_rejected(client) -> None:
    """FastAPI's Form(...) layer rejects empty/missing `q` with 422 before
    our handler runs. Confirms the endpoint enforces a non-empty body."""
    tc, _ = client
    with tc as c:
        resp = c.post("/v2/colabfold/plmmsa/ticket/msa", data={"q": ""})
    assert resp.status_code == 422


def test_ticket_msa_fasta_without_sequence_is_400(client) -> None:
    """A FASTA record with only header (no sequence lines) makes it past
    FastAPI's Form layer and hits our parser, which 400s."""
    tc, _ = client
    with tc as c:
        resp = c.post("/v2/colabfold/plmmsa/ticket/msa", data={"q": ">header-only\n"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "E_INVALID_FASTA"
