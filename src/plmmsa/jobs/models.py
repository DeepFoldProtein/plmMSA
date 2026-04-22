from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResult(BaseModel):
    format: str = Field("a3m", description="Wire format of `payload` (a3m, stockholm, ...).")
    payload: str = Field(..., description="The MSA serialized in the declared format.")
    stats: dict[str, Any] = Field(default_factory=dict)


class JobError(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] | None = None


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    request: dict[str, Any]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: JobResult | None = None
    error: JobError | None = None

    def is_terminal(self) -> bool:
        return self.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
