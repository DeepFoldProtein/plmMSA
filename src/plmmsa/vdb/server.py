from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from plmmsa import __version__
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.vdb.base import VDB
from plmmsa.vdb.registry import load_enabled_collections

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    collection: str = Field(..., description="VDB collection id (e.g. 'ankh_uniref50').")
    vectors: list[list[float]] = Field(..., min_length=1)
    k: int = Field(100, ge=1, le=10_000)
    nprobe: int | None = Field(None, ge=1, le=10_000)


class NeighborResponse(BaseModel):
    id: str
    distance: float


class SearchResponse(BaseModel):
    collection: str
    k: int
    results: list[list[NeighborResponse]]


class CollectionStatus(BaseModel):
    loaded: bool
    dim: int
    model_backend: str


class HealthResponse(BaseModel):
    status: str
    service: str
    collections: dict[str, CollectionStatus]


def create_app(*, collections_override: dict[str, VDB] | None = None) -> FastAPI:
    """Factory for the VDB FastAPI app.

    Tests pass `collections_override` to avoid loading real multi-GB FAISS
    indexes. Production uses the default: `load_enabled_collections(settings)`.
    """
    settings = get_settings()
    app = FastAPI(
        title="plmMSA-vdb",
        version=__version__,
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
    )

    collections = (
        dict(collections_override)
        if collections_override is not None
        else load_enabled_collections(settings)
    )
    logger.info("vdb server: collections loaded = %s", sorted(collections.keys()))

    @app.exception_handler(PlmMSAError)
    async def _err(_: Request, exc: PlmMSAError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        statuses = {
            name: CollectionStatus(loaded=True, dim=c.dim, model_backend=c.model_backend)
            for name, c in collections.items()
        }
        return HealthResponse(status="ok", service="vdb", collections=statuses)

    @app.post("/search", response_model=SearchResponse, tags=["vdb"])
    async def search(req: SearchRequest) -> SearchResponse:
        collection = collections.get(req.collection)
        if collection is None:
            raise PlmMSAError(
                f"Collection '{req.collection}' is not loaded.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={
                    "requested": req.collection,
                    "available": sorted(collections.keys()),
                },
            )

        arr = np.asarray(req.vectors, dtype=np.float32)
        try:
            hits = collection.search(arr, k=req.k, nprobe=req.nprobe)
        except ValueError as exc:
            raise PlmMSAError(
                str(exc),
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
                detail={"collection": req.collection, "shape": list(arr.shape)},
            ) from exc

        return SearchResponse(
            collection=req.collection,
            k=req.k,
            results=[[NeighborResponse(id=n.id, distance=n.distance) for n in row] for row in hits],
        )

    return app
