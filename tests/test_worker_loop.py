from __future__ import annotations

import pytest
from fakeredis import FakeAsyncRedis

from plmmsa.jobs import JobStatus, JobStore
from plmmsa.jobs.models import JobResult
from plmmsa.worker.__main__ import _one_iteration


class _StubOrchestrator:
    def __init__(self, result_payload: str = ">query\nMKT\n") -> None:
        self._result = JobResult(format="a3m", payload=result_payload, stats={"pipeline": "stub"})
        self.calls: list[dict] = []

    async def run(self, request: dict) -> JobResult:
        self.calls.append(request)
        return self._result


class _BoomOrchestrator:
    async def run(self, request: dict) -> JobResult:
        raise RuntimeError("synthetic pipeline crash")


class _CancelDuringRunOrchestrator:
    def __init__(self, store: JobStore, job_id_ref: dict) -> None:
        self.store = store
        self.job_id_ref = job_id_ref

    async def run(self, request: dict) -> JobResult:
        await self.store.cancel(self.job_id_ref["id"])
        return JobResult(format="a3m", payload=">query\nMKT\n")


@pytest.fixture
async def store() -> JobStore:
    return JobStore(FakeAsyncRedis())


async def test_iteration_processes_queued_job(store: JobStore) -> None:
    orchestrator = _StubOrchestrator()
    job = await store.create({"sequences": ["MKT"], "model": "ankh_cl"})

    did_work = await _one_iteration(store, orchestrator)  # pyright: ignore[reportArgumentType]
    assert did_work is True

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.SUCCEEDED
    assert fetched.result is not None
    assert ">query" in fetched.result.payload


async def test_iteration_returns_false_on_empty_queue(store: JobStore) -> None:
    orchestrator = _StubOrchestrator()
    did_work = await _one_iteration(store, orchestrator)  # pyright: ignore[reportArgumentType]
    assert did_work is False


async def test_iteration_marks_failed_on_pipeline_exception(store: JobStore) -> None:
    orchestrator = _BoomOrchestrator()
    job = await store.create({"sequences": ["MKT"]})

    await _one_iteration(store, orchestrator)  # pyright: ignore[reportArgumentType]

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.FAILED
    assert fetched.error is not None
    assert fetched.error.code == "E_INTERNAL"
    assert "synthetic" in fetched.error.message


async def test_iteration_respects_cancellation_during_run(store: JobStore) -> None:
    job_ref: dict = {}
    orchestrator = _CancelDuringRunOrchestrator(store, job_ref)
    job = await store.create({"sequences": ["MKT"]})
    job_ref["id"] = job.id

    await _one_iteration(store, orchestrator)  # pyright: ignore[reportArgumentType]

    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.CANCELLED
    assert fetched.result is None
