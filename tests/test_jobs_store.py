from __future__ import annotations

import pytest
from fakeredis import FakeAsyncRedis

from plmmsa.jobs import JobError, JobResult, JobStatus, JobStore


@pytest.fixture
async def store() -> JobStore:
    redis = FakeAsyncRedis()
    return JobStore(redis)


async def test_create_returns_queued_job(store: JobStore) -> None:
    job = await store.create({"sequences": ["MKT"]})

    assert job.status is JobStatus.QUEUED
    assert job.request == {"sequences": ["MKT"]}
    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.QUEUED


async def test_claim_next_marks_running(store: JobStore) -> None:
    created = await store.create({"sequences": ["MKT"]})
    claimed = await store.claim_next(timeout_s=0.1)
    assert claimed is not None
    assert claimed.id == created.id
    assert claimed.status is JobStatus.RUNNING
    assert claimed.started_at is not None


async def test_claim_next_times_out_on_empty_queue(store: JobStore) -> None:
    result = await store.claim_next(timeout_s=0.1)
    assert result is None


async def test_mark_succeeded_stores_result(store: JobStore) -> None:
    job = await store.create({"sequences": ["MKT"]})
    await store.mark_running(job.id)
    await store.mark_succeeded(
        job.id,
        JobResult(format="a3m", payload=">query\nMKT\n", stats={"depth": 1}),
    )

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.SUCCEEDED
    assert fetched.result is not None
    assert fetched.result.payload.startswith(">query")
    assert fetched.finished_at is not None


async def test_mark_failed_stores_error(store: JobStore) -> None:
    job = await store.create({"sequences": ["MKT"]})
    await store.mark_failed(job.id, JobError(code="E_INTERNAL", message="boom"))

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.FAILED
    assert fetched.error is not None
    assert fetched.error.code == "E_INTERNAL"


async def test_cancel_idempotent_on_terminal(store: JobStore) -> None:
    job = await store.create({"sequences": ["MKT"]})
    await store.mark_succeeded(job.id, JobResult(format="a3m", payload=""))

    cancelled = await store.cancel(job.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.SUCCEEDED  # terminal state preserved


async def test_cancel_flips_queued(store: JobStore) -> None:
    job = await store.create({"sequences": ["MKT"]})
    cancelled = await store.cancel(job.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.finished_at is not None


async def test_get_unknown_returns_none(store: JobStore) -> None:
    assert await store.get("does-not-exist") is None
