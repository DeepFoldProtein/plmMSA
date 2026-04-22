from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from plmmsa import __version__
from plmmsa.align.base import Aligner, Alignment, AlignMode
from plmmsa.align.registry import load_enabled_aligners
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError

logger = logging.getLogger(__name__)


class AlignRequest(BaseModel):
    aligner: str = Field("plmalign", description="Aligner id (e.g. 'plmalign').")
    mode: AlignMode = Field("local", description="'local' or 'global'.")
    query_embedding: list[list[float]] = Field(..., min_length=1)
    target_embeddings: list[list[list[float]]] = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class AlignmentResponse(BaseModel):
    score: float
    mode: str
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    columns: list[list[int]]


class AlignResponse(BaseModel):
    aligner: str
    mode: str
    alignments: list[AlignmentResponse]


class AlignerStatus(BaseModel):
    loaded: bool
    display_name: str


class HealthResponse(BaseModel):
    status: str
    service: str
    aligners: dict[str, AlignerStatus]


def _to_response(a: Alignment) -> AlignmentResponse:
    return AlignmentResponse(
        score=a.score,
        mode=a.mode,
        query_start=a.query_start,
        query_end=a.query_end,
        target_start=a.target_start,
        target_end=a.target_end,
        columns=[[qi, ti] for qi, ti in a.columns],
    )


def create_app(*, aligners_override: dict[str, Aligner] | None = None) -> FastAPI:
    """Factory for the align FastAPI app.

    Tests pass `aligners_override` to inject stub aligners.
    """
    settings = get_settings()
    app = FastAPI(
        title="plmMSA-align",
        version=__version__,
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
    )

    aligners = (
        dict(aligners_override)
        if aligners_override is not None
        else load_enabled_aligners(settings)
    )
    logger.info("align server: aligners loaded = %s", sorted(aligners.keys()))

    @app.exception_handler(PlmMSAError)
    async def _err(_: Request, exc: PlmMSAError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        statuses = {
            name: AlignerStatus(loaded=True, display_name=a.display_name)
            for name, a in aligners.items()
        }
        return HealthResponse(status="ok", service="align", aligners=statuses)

    @app.post("/align", response_model=AlignResponse, tags=["align"])
    async def align(req: AlignRequest) -> AlignResponse:
        aligner = aligners.get(req.aligner)
        if aligner is None:
            raise PlmMSAError(
                f"Aligner '{req.aligner}' is not loaded.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={"requested": req.aligner, "available": sorted(aligners.keys())},
            )

        query = np.asarray(req.query_embedding, dtype=np.float32)
        targets = [np.asarray(t, dtype=np.float32) for t in req.target_embeddings]

        if query.ndim != 2:
            raise PlmMSAError(
                "query_embedding must be a 2-D array [Lq, D].",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
                detail={"shape": list(query.shape)},
            )
        for idx, t in enumerate(targets):
            if t.ndim != 2 or t.shape[1] != query.shape[1]:
                raise PlmMSAError(
                    f"target_embeddings[{idx}] shape {list(t.shape)} "
                    f"incompatible with query dim {query.shape[1]}.",
                    code=ErrorCode.INVALID_FASTA,
                    http_status=400,
                    detail={"index": idx, "shape": list(t.shape)},
                )

        alignments = aligner.align(query, targets, mode=req.mode, **req.options)
        return AlignResponse(
            aligner=req.aligner,
            mode=req.mode,
            alignments=[_to_response(a) for a in alignments],
        )

    return app
