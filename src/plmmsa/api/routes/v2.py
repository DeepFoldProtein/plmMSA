from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Path
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from plmmsa import __version__
from plmmsa.api.auth import require_admin_token
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.jobs import JobStore

router = APIRouter(tags=["v2"])

_job_store: JobStore | None = None


async def _get_job_store() -> JobStore:
    """Lazy singleton — production uses CACHE_URL; tests override via
    `plmmsa.api.routes.v2._job_store = <fake>`."""
    global _job_store
    if _job_store is None:
        cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
        _job_store = JobStore(Redis.from_url(cache_url, decode_responses=False))
    return _job_store


class VersionResponse(BaseModel):
    plmmsa: str
    api: str
    models: dict[str, dict[str, Any]]


@router.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    settings = get_settings()
    models = {
        name: getattr(settings.models, name).model_dump()
        for name in ("ankh_cl", "ankh_large", "esm1b", "prott5")
    }
    return VersionResponse(plmmsa=__version__, api="v2", models=models)


class SubmitRequest(BaseModel):
    sequences: list[str] = Field(..., description="One string per chain.", min_length=1)
    model: str = Field("ankh_cl", description="PLM backend id.")
    output_format: str = Field("a3m", description="Wire format: a3m | stockholm | ...")
    paired: bool = Field(False, description="Produce paired MSA across chains.")
    # Orchestrator knobs (optional; the orchestrator supplies defaults).
    query_id: str | None = Field(None, description="Shown in the A3M query header.")
    collection: str | None = Field(
        None,
        description="VDB collection id; defaults to `<model>_uniref50`.",
    )
    k: int | None = Field(
        None, ge=1, le=10_000, description="Number of FAISS neighbors to fetch."
    )
    aligner: str | None = Field(None, description="Aligner id (default: plmalign).")
    mode: str | None = Field(None, description="local | global.")
    options: dict[str, Any] | None = Field(
        None, description="Pass-through aligner kwargs (gap_open, gap_extend, ...)."
    )


class SubmitResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


@router.post("/msa", status_code=202, response_model=SubmitResponse)
async def submit_msa(req: SubmitRequest) -> SubmitResponse:
    store = await _get_job_store()
    job = await store.create(req.model_dump())
    return SubmitResponse(
        job_id=job.id,
        status=job.status.value,
        status_url=f"/v2/msa/{job.id}",
    )


@router.get("/msa/{job_id}")
async def get_msa(job_id: str = Path(..., min_length=1)) -> JSONResponse:
    store = await _get_job_store()
    job = await store.get(job_id)
    if job is None:
        raise PlmMSAError(
            f"Job {job_id} not found.",
            code=ErrorCode.JOB_NOT_FOUND,
            http_status=404,
            detail={"job_id": job_id},
        )
    return JSONResponse(status_code=200, content=job.model_dump(mode="json"))


@router.delete("/msa/{job_id}", status_code=204)
async def cancel_msa(job_id: str = Path(..., min_length=1)) -> Response:
    store = await _get_job_store()
    job = await store.cancel(job_id)
    if job is None:
        raise PlmMSAError(
            f"Job {job_id} not found.",
            code=ErrorCode.JOB_NOT_FOUND,
            http_status=404,
            detail={"job_id": job_id},
        )
    return Response(status_code=204)


async def _forward(
    target: str, path: str, payload: dict, *, timeout: float = 120.0
) -> JSONResponse:
    """Common httpx passthrough used by /v2/embed and /v2/search."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.post(f"{target}{path}", json=payload)
    except httpx.HTTPError as exc:
        raise PlmMSAError(
            f"Upstream {target} unreachable.",
            code=ErrorCode.INTERNAL,
            http_status=502,
            detail={"cause": str(exc)[:200]},
        ) from exc

    try:
        body = upstream.json()
    except ValueError as exc:
        raise PlmMSAError(
            f"Upstream {target} returned non-JSON.",
            code=ErrorCode.INTERNAL,
            http_status=502,
            detail={"status": upstream.status_code},
        ) from exc

    return JSONResponse(status_code=upstream.status_code, content=body)


class EmbedRequest(BaseModel):
    model: str = Field("ankh_cl", description="PLM backend id.")
    sequences: list[str] = Field(..., min_length=1)


@router.post("/embed", dependencies=[Depends(require_admin_token)])
async def embed(req: EmbedRequest) -> JSONResponse:
    """Forward raw per-residue embeddings from the embedding service.

    Bearer-token gated (see `plmmsa.api.auth.require_admin_token`). Per-token
    rate limits and Redis-backed token records are a future milestone.
    """
    target = os.environ.get("EMBEDDING_URL", "http://embedding:8081")
    return await _forward(target, "/embed", req.model_dump())


class SearchRequest(BaseModel):
    collection: str = Field(..., description="VDB collection id.")
    vectors: list[list[float]] = Field(..., min_length=1)
    k: int = Field(100, ge=1, le=10_000)
    nprobe: int | None = Field(None, ge=1, le=10_000)


@router.post("/search", dependencies=[Depends(require_admin_token)])
async def search(req: SearchRequest) -> JSONResponse:
    """Forward a vector search to the VDB service.

    Same auth model as `/v2/embed`: bearer token against `ADMIN_TOKEN` for now.
    """
    target = os.environ.get("VDB_URL", "http://vdb:8082")
    return await _forward(target, "/search", req.model_dump())


class AlignRequest(BaseModel):
    aligner: str = Field("plmalign", description="Aligner id.")
    mode: str = Field("local", description="'local' or 'global'.")
    query_embedding: list[list[float]] = Field(..., min_length=1)
    target_embeddings: list[list[list[float]]] = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/align", dependencies=[Depends(require_admin_token)])
async def align(req: AlignRequest) -> JSONResponse:
    """Forward a pairwise alignment to the align service.

    Same auth model as `/v2/embed` and `/v2/search`. Aligner-specific
    tunables (gap_open, gap_extend, normalize, ...) ride in `options`.
    """
    target = os.environ.get("ALIGN_URL", "http://align:8083")
    return await _forward(target, "/align", req.model_dump())
