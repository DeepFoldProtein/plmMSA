from __future__ import annotations

import gc
import logging
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from plmmsa import __version__
from plmmsa.align import binary as align_binary
from plmmsa.align.base import Aligner, Alignment, AlignMode
from plmmsa.align.registry import load_enabled_aligners
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError

logger = logging.getLogger(__name__)


def _cleanup_cuda(reason: str) -> None:
    """Best-effort CUDA allocator cleanup; never mask the original failure."""
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            logger.debug("align: torch.cuda.ipc_collect() failed", exc_info=True)
        logger.info("align: CUDA cache cleanup complete (%s)", reason)
    except Exception:
        logger.exception("align: CUDA cache cleanup failed (%s)", reason)


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


def _maybe_register_gpu_builders(settings: Any) -> None:
    """Swap numpy builders for torch-on-device versions if any enabled
    aligner requests a non-CPU device in settings. Idempotent.

    Runs at align-service startup only — other services don't need score
    matrices. Missing torch/CUDA logs a WARN and leaves the numpy builders
    in place; the operator chose the knob and is responsible for wiring
    GPU access into this container.
    """
    devices: set[str] = set()
    for name in ("plmalign", "plm_blast"):
        entry = getattr(settings.aligners, name, None)
        if entry is None:
            continue
        dev = getattr(entry, "score_matrix_device", "cpu") or "cpu"
        if dev != "cpu":
            devices.add(dev)
    if not devices:
        return
    if len(devices) > 1:
        logger.warning(
            "align: multiple score_matrix_device values requested %s; "
            "using the first — operator must reconcile settings",
            sorted(devices),
        )
    device = sorted(devices)[0]
    try:
        from plmmsa.align import torch_score_matrix

        torch_score_matrix.register_all(device=device)
        logger.info("align: registered torch score-matrix builders on %s", device)
    except Exception:
        logger.exception(
            "align: failed to register torch builders on %s — falling back to numpy",
            device,
        )


def _warmup_fused_sinkhorn(settings: Any) -> None:
    """One-shot call through `unbalanced_sinkhorn_flash` so torch.compile
    pays the graph-build cost at startup, not on the first real job.

    Skipped when OTalign's `fused_sinkhorn` is off, when torch isn't
    available, or when no CUDA device is visible (the flash path falls
    back to eager on CPU anyway).
    """
    cfg = getattr(settings.aligners, "otalign", None)
    if cfg is None or not getattr(cfg, "fused_sinkhorn", False):
        return
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            logger.info("align: fused_sinkhorn requested but CUDA unavailable — skipping warmup")
            return

        from plmmsa.align.sinkhorn_flash import unbalanced_sinkhorn_flash

        # Matrix size is representative of the CASP15 target distribution;
        # torch.compile with dynamic=True reuses this graph for other
        # shapes in the same rank family.
        device = cfg.device or "cuda:0"
        c = torch.randn(96, 256, dtype=torch.float32, device=device)
        unbalanced_sinkhorn_flash(
            c,
            eps=0.1,
            tau=1.0,
            n_iter=100,
            tol=1e-4,
        )
        _ = np  # keep numpy import side-effect free
        logger.info(
            "align: FlashSinkhorn warmup complete (device=%s)",
            device,
        )
    except Exception:
        logger.exception(
            "align: FlashSinkhorn warmup failed — first real job will pay compile cost"
        )


def create_app(*, aligners_override: dict[str, Aligner] | None = None) -> FastAPI:
    """Factory for the align FastAPI app.

    Tests pass `aligners_override` to inject stub aligners.
    """
    settings = get_settings()
    empty_cache_after = os.environ.get("PLMMSA_ALIGN_CUDA_EMPTY_CACHE_AFTER_REQUEST", "")
    empty_cache_after_request = empty_cache_after.lower() in {"1", "true", "yes", "on"}
    if empty_cache_after_request:
        logger.info(
            "align: CUDA cache cleanup after every /align request is ENABLED "
            "(PLMMSA_ALIGN_CUDA_EMPTY_CACHE_AFTER_REQUEST)"
        )
    _maybe_register_gpu_builders(settings)
    app = FastAPI(
        title="plmMSA-align",
        version=__version__,
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
    )
    from plmmsa.request_context import RequestContextMiddleware

    app.add_middleware(RequestContextMiddleware, service="align")

    from plmmsa.metrics import MetricsMiddleware
    from plmmsa.metrics import router as metrics_router

    app.add_middleware(MetricsMiddleware, service="align")
    app.include_router(metrics_router)

    aligners = (
        dict(aligners_override)
        if aligners_override is not None
        else load_enabled_aligners(settings)
    )
    logger.info("align server: aligners loaded = %s", sorted(aligners.keys()))
    _warmup_fused_sinkhorn(settings)

    @app.exception_handler(PlmMSAError)
    async def _err(request: Request, exc: PlmMSAError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.exception(
                "align: %s %s → %d %s: %s",
                request.method,
                request.url.path,
                exc.http_status,
                exc.code.value if hasattr(exc.code, "value") else exc.code,
                exc.message,
            )
        else:
            logger.warning(
                "align: %s %s → %d %s: %s",
                request.method,
                request.url.path,
                exc.http_status,
                exc.code.value if hasattr(exc.code, "value") else exc.code,
                exc.message,
            )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
        )

    @app.middleware("http")
    async def _cuda_cleanup_middleware(request: Request, call_next: Any) -> Response:
        gpu_endpoint = request.url.path.startswith("/align")
        try:
            response = await call_next(request)
        except Exception:
            if gpu_endpoint:
                _cleanup_cuda(f"{request.url.path}:exception")
            raise
        if gpu_endpoint and (empty_cache_after_request or response.status_code >= 500):
            _cleanup_cuda(f"{request.url.path}:status={response.status_code}")
        return response

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

        # Server-side defaults from settings fill in kwargs the client didn't
        # specify. Client-supplied `options` still win on a per-request basis,
        # so operators can change the default without breaking power users.
        merged_options = _settings_defaults_for(req.aligner, settings) | dict(req.options)

        alignments = aligner.align(query, targets, mode=req.mode, **merged_options)
        return AlignResponse(
            aligner=req.aligner,
            mode=req.mode,
            alignments=[_to_response(a) for a in alignments],
        )

    @app.post("/align/bin", tags=["align"])
    async def align_bin(request: Request) -> JSONResponse:
        """Binary-input variant of /align.

        Accepts a `application/x-plmmsa-align` framed body (see
        `plmmsa.align.binary`) that carries metadata + query + targets
        as a compact byte stream. Response is the same JSON as /align —
        alignment results are small. Worker uses this by default to
        avoid the JSON parse bottleneck on 500x400x1024 f32 bodies.
        """
        body = await request.body()
        try:
            metadata, query, targets = align_binary.decode(body)
        except ValueError as exc:
            raise PlmMSAError(
                f"Invalid binary align frame: {exc}",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
                detail={"cause": str(exc)[:200]},
            ) from exc

        aligner_id = str(metadata.get("aligner") or "plmalign")
        mode: AlignMode = metadata.get("mode") or "local"  # pyright: ignore
        options = metadata.get("options") or {}

        aligner = aligners.get(aligner_id)
        if aligner is None:
            raise PlmMSAError(
                f"Aligner '{aligner_id}' is not loaded.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={"requested": aligner_id, "available": sorted(aligners.keys())},
            )
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

        merged_options = _settings_defaults_for(aligner_id, settings) | dict(options)
        alignments = aligner.align(query, targets, mode=mode, **merged_options)
        return JSONResponse(
            status_code=200,
            content=AlignResponse(
                aligner=aligner_id,
                mode=mode,
                alignments=[_to_response(a) for a in alignments],
            ).model_dump(mode="json"),
        )

    return app


def _settings_defaults_for(aligner_id: str, settings: Any) -> dict[str, Any]:
    """Pull aligner-specific defaults from settings.

    Each aligner has its own set of accepted kwargs. We only surface the
    ones the aligner knows; anything else would leak through as a
    `TypeError` on `aligner.align(**opts)`.
    """
    if aligner_id == "plmalign":
        cfg = settings.aligners.plmalign
        return {
            "score_matrix": cfg.score_matrix,
            "gap_open": cfg.gap_open,
            "gap_extend": cfg.gap_extend,
        }
    if aligner_id == "plm_blast":
        cfg = settings.aligners.plm_blast
        return {
            "score_matrix": cfg.score_matrix,
            "gap_penalty": cfg.gap_penalty,
            "min_span": cfg.min_span,
            "window_size": cfg.window_size,
            "sigma_factor": cfg.sigma_factor,
            "border_stride": cfg.border_stride,
        }
    if aligner_id == "otalign":
        cfg = settings.aligners.otalign
        # Only forward `device` if explicitly set — empty string lets
        # OTalign's auto-detect kick in.
        out: dict[str, Any] = {}
        if cfg.device:
            out["device"] = cfg.device
        # FlashSinkhorn opt-in lives in settings; the aligner picks it
        # up through the `fused_sinkhorn` kwarg in align().
        if cfg.fused_sinkhorn:
            out["fused_sinkhorn"] = True
        return out
    return {}
