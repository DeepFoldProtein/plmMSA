from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from plmmsa import __version__
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.plm.base import PLM
from plmmsa.plm.registry import load_enabled_backends

logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    model: str = Field(..., description="PLM backend id (e.g. 'ankh_cl').")
    sequences: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[list[float]]]


class ModelStatus(BaseModel):
    loaded: bool
    device: str
    dim: int


class HealthResponse(BaseModel):
    status: str
    service: str
    models: dict[str, ModelStatus]


def create_app(*, backends_override: dict[str, PLM] | None = None) -> FastAPI:
    """Factory for the embedding FastAPI app.

    Tests pass `backends_override` to skip real model loading.
    """
    settings = get_settings()
    app = FastAPI(
        title="plmMSA-embedding",
        version=__version__,
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
    )

    if backends_override is not None:
        backends: dict[str, PLM] = dict(backends_override)
    else:
        backends = load_enabled_backends(settings)

    logger.info("embedding server: backends loaded = %s", sorted(backends.keys()))

    @app.exception_handler(PlmMSAError)
    async def _err(_: Request, exc: PlmMSAError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        models = {
            name: ModelStatus(loaded=True, device=str(b.device), dim=b.dim)
            for name, b in backends.items()
        }
        return HealthResponse(status="ok", service="embedding", models=models)

    @app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
    async def embed(req: EmbedRequest) -> EmbedResponse:
        backend = backends.get(req.model)
        if backend is None:
            raise PlmMSAError(
                f"Model '{req.model}' is not loaded.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={"requested": req.model, "available": sorted(backends.keys())},
            )

        try:
            tensors = backend.encode(req.sequences)
        except Exception as exc:
            logger.exception("embed failed for model=%s", req.model)
            if _is_cuda_oom(exc):
                raise PlmMSAError(
                    "GPU out of memory while encoding.",
                    code=ErrorCode.GPU_OOM,
                    http_status=503,
                    detail={"model": req.model, "cause": str(exc)[:200]},
                ) from exc
            raise

        return EmbedResponse(
            model=req.model,
            dim=backend.dim,
            embeddings=[t.tolist() for t in tensors],
        )

    return app


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


# Module-level `app` is NOT created here — use `create_app` via uvicorn's
# --factory flag (see `plmmsa/embedding/__main__.py`) so tests and dry-runs do
# not trigger multi-GB model downloads just to import the module.
def _unused(_: Any) -> None: ...
