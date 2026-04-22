from __future__ import annotations

import time
import uuid
from typing import Any

from redis.asyncio import Redis

from plmmsa.jobs.models import Job, JobError, JobResult, JobStatus

_JOB_KEY_PREFIX = "plmmsa:job:"
_QUEUE_KEY = "plmmsa:queue"


def _job_key(job_id: str) -> str:
    return f"{_JOB_KEY_PREFIX}{job_id}"


class JobStore:
    """Minimal Redis-backed job store.

    Records live as JSON strings under `plmmsa:job:{id}`; the pending queue is
    a Redis list `plmmsa:queue` holding job ids. Callers that want richer
    semantics (priorities, retries, distributed locking) should replace this
    with arq / rq / a proper broker — this shape is deliberately small so the
    API + worker stay obvious.
    """

    def __init__(self, redis: Redis, *, queue_key: str = _QUEUE_KEY) -> None:
        self.redis = redis
        self.queue_key = queue_key

    # --- Writes -----------------------------------------------------------

    async def create(self, request: dict[str, Any]) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            status=JobStatus.QUEUED,
            request=request,
            created_at=time.time(),
        )
        await self._save(job)
        await self.redis.rpush(self.queue_key, job.id)  # pyright: ignore[reportGeneralTypeIssues]
        return job

    async def mark_running(self, job_id: str) -> Job | None:
        job = await self.get(job_id)
        if job is None:
            return None
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        await self._save(job)
        return job

    async def mark_succeeded(self, job_id: str, result: JobResult) -> Job | None:
        job = await self.get(job_id)
        if job is None:
            return None
        job.status = JobStatus.SUCCEEDED
        job.finished_at = time.time()
        job.result = result
        await self._save(job)
        return job

    async def mark_failed(self, job_id: str, error: JobError) -> Job | None:
        job = await self.get(job_id)
        if job is None:
            return None
        job.status = JobStatus.FAILED
        job.finished_at = time.time()
        job.error = error
        await self._save(job)
        return job

    async def cancel(self, job_id: str) -> Job | None:
        """Flip a queued/running job to `cancelled`. Idempotent on terminal jobs."""
        job = await self.get(job_id)
        if job is None:
            return None
        if job.is_terminal():
            return job
        job.status = JobStatus.CANCELLED
        job.finished_at = time.time()
        await self._save(job)
        return job

    # --- Reads ------------------------------------------------------------

    async def get(self, job_id: str) -> Job | None:
        raw = await self.redis.get(_job_key(job_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return Job.model_validate_json(raw)

    # --- Worker-side ------------------------------------------------------

    async def claim_next(self, *, timeout_s: float = 1.0) -> Job | None:
        """Block (up to `timeout_s`) for the next queued job, transition to
        running, and return it. Returns None on timeout or if the popped
        record has gone missing / been cancelled in the meantime."""
        # redis-py's async blpop returns an Awaitable but its type stubs lie —
        # hence the explicit cast on the result shape below.
        popped = await self.redis.blpop([self.queue_key], timeout=timeout_s)  # pyright: ignore[reportGeneralTypeIssues]
        if popped is None:
            return None
        _, raw_id = popped
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8")
        job = await self.get(raw_id)
        if job is None or job.status == JobStatus.CANCELLED:
            return None
        return await self.mark_running(raw_id)

    # --- Internal ---------------------------------------------------------

    async def _save(self, job: Job) -> None:
        await self.redis.set(_job_key(job.id), job.model_dump_json())
