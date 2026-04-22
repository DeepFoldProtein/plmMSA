from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ErrorCode(StrEnum):
    SEQ_TOO_LONG = "E_SEQ_TOO_LONG"
    TOO_MANY_CHAINS = "E_TOO_MANY_CHAINS"
    INVALID_FASTA = "E_INVALID_FASTA"
    UNSUPPORTED_MODEL = "E_UNSUPPORTED_MODEL"
    UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"
    GPU_OOM = "E_GPU_OOM"
    UNIREF_MISS = "E_UNIREF_MISS"
    QUEUE_FULL = "E_QUEUE_FULL"
    JOB_NOT_FOUND = "E_JOB_NOT_FOUND"
    CANCELLED = "E_CANCELLED"
    AUTH_MISSING = "E_AUTH_MISSING"
    AUTH_INVALID = "E_AUTH_INVALID"
    AUTH_FORBIDDEN = "E_AUTH_FORBIDDEN"
    RATE_LIMITED = "E_RATE_LIMITED"
    NOT_IMPLEMENTED = "E_NOT_IMPLEMENTED"
    GONE = "E_GONE"
    INTERNAL = "E_INTERNAL"


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    detail: dict[str, Any] | None = None


class PlmMSAError(Exception):
    http_status: int = 500
    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        http_status: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status

    def as_response(self) -> ErrorResponse:
        return ErrorResponse(code=self.code, message=self.message, detail=self.detail)
