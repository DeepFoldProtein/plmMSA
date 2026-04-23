from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from plmmsa import __version__
from plmmsa.config import get_settings
from plmmsa.embedding.shard_store import ShardStore
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


class EmbedByIdRequest(BaseModel):
    model: str = Field(..., description="PLM backend id with a configured shard store.")
    ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description=(
            "UniRef50 ids (e.g. 'UniRef50_Q9RYE6'). Orchestrator chunks "
            "its own calls to this endpoint, but the cap here protects "
            "external callers from accidental JSON-blowup."
        ),
    )


class EmbedByIdResponse(BaseModel):
    model: str
    dim: int
    found: dict[str, list[list[float]]]
    missing: list[str]


class ModelStatus(BaseModel):
    loaded: bool
    device: str
    dim: int


class HealthResponse(BaseModel):
    status: str
    service: str
    models: dict[str, ModelStatus]


def _build_shard_stores(settings: Any) -> dict[str, ShardStore]:
    """Instantiate a ShardStore per model that has `shard_root` set.

    Deliberately independent of `load_enabled_backends` — a model with
    `enabled = false` can still have its shards served if the operator
    wants the cache-only behavior. Missing roots produce a WARN (inside
    the ShardStore constructor) and a registered-but-empty store; the
    fetch path returns all-misses for them.
    """
    out: dict[str, ShardStore] = {}
    # Shared Redis path-index URL. When set, the ShardStore uses MGET
    # instead of SQLite for filename → folder lookups. Populate once
    # via `python -m plmmsa.tools.build_shard_index`.
    shard_redis_url = os.environ.get("PLMMSA_SHARD_INDEX_REDIS_URL", "").strip() or None
    for name in ("ankh_cl", "ankh_large", "esm1b", "prott5"):
        cfg = getattr(settings.models, name)
        if not cfg.shard_root:
            continue
        try:
            out[name] = ShardStore(
                root=cfg.shard_root,
                index_db=cfg.shard_index,
                fallback_dirs=tuple(cfg.shard_fallback_dirs),
                dim=cfg.shard_dim or 1024,
                redis_url=shard_redis_url,
                redis_key_prefix=f"shard:{name}:",
            )
        except Exception:
            logger.exception("shard_store: failed to register %s", name)
    return out


def create_app(
    *,
    backends_override: dict[str, PLM] | None = None,
    shard_stores_override: dict[str, ShardStore] | None = None,
) -> FastAPI:
    """Factory for the embedding FastAPI app.

    Tests pass `backends_override` to skip real model loading, and
    `shard_stores_override` to pin a fake shard reader.
    """
    settings = get_settings()

    if backends_override is not None:
        backends: dict[str, PLM] = dict(backends_override)
    else:
        backends = load_enabled_backends(settings)

    if shard_stores_override is not None:
        shard_stores: dict[str, ShardStore] = dict(shard_stores_override)
    else:
        shard_stores = _build_shard_stores(settings)

    cache_url = os.environ.get("PLMMSA_EMBEDDING_CACHE_URL")
    cache_client: Any | None = None
    cache_ttl_s = int(os.environ.get("PLMMSA_EMBEDDING_CACHE_TTL_S", str(60 * 60 * 24 * 30)))
    if cache_url:
        from redis.asyncio import Redis as _Redis

        cache_client = _Redis.from_url(cache_url, decode_responses=False)
        logger.info("embedding cache enabled at %s (ttl %ds)", cache_url, cache_ttl_s)

    empty_cache_after = settings.embedding.empty_cache_after_request
    if empty_cache_after:
        logger.info(
            "embedding: torch.cuda.empty_cache() after every /embed is ENABLED "
            "(see settings.embedding.empty_cache_after_request)"
        )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """On SIGTERM, clear GPU caches so the next container start finds a
        clean device. Uvicorn handles the drain; we just release the
        resources model wrappers hold onto."""
        yield
        try:
            import torch as _torch  # heavy import; lazy

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                logger.info("embedding: released CUDA cache on shutdown")
        except Exception:
            logger.exception("embedding: GPU cache release failed on shutdown")

    app = FastAPI(
        title="plmMSA-embedding",
        version=__version__,
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
        lifespan=_lifespan,
    )

    logger.info("embedding server: backends loaded = %s", sorted(backends.keys()))
    logger.info("embedding server: shard stores = %s", sorted(shard_stores.keys()))

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

    async def _resolve_embeddings(
        model: str, sequences: list[str]
    ) -> tuple[list[Any], int]:
        """Cache-aware embedding resolution shared by `/embed` and
        `/embed/bin`. Returns `(per_seq_tensors_in_input_order, dim)`.
        Each item is either a `torch.Tensor` (fresh encode, CPU) or
        whatever the cache unpickled (also a torch.Tensor in practice).
        """
        backend = backends.get(model)
        if backend is None:
            raise PlmMSAError(
                f"Model '{model}' is not loaded.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={"requested": model, "available": sorted(backends.keys())},
            )

        resolved: list[Any] = [None] * len(sequences)
        miss_positions: list[int] = []
        miss_seqs: list[str] = []
        if cache_client is not None:
            keys = [_cache_key(model, s) for s in sequences]
            cached = await cache_client.mget(keys)
            for i, blob in enumerate(cached):
                if blob is None:
                    miss_positions.append(i)
                    miss_seqs.append(sequences[i])
                else:
                    try:
                        resolved[i] = pickle.loads(blob)
                    except Exception:
                        logger.exception("cache decode failed; re-encoding")
                        miss_positions.append(i)
                        miss_seqs.append(sequences[i])
        else:
            miss_positions = list(range(len(sequences)))
            miss_seqs = list(sequences)

        if miss_seqs:
            try:
                tensors = backend.encode(miss_seqs)
            except Exception as exc:
                logger.exception("embed failed for model=%s", model)
                if _is_cuda_oom(exc):
                    raise PlmMSAError(
                        "GPU out of memory while encoding.",
                        code=ErrorCode.GPU_OOM,
                        http_status=503,
                        detail={"model": model, "cause": str(exc)[:200]},
                    ) from exc
                raise
            for pos, arr in zip(miss_positions, tensors, strict=True):
                resolved[pos] = arr
            if cache_client is not None:
                pipe = cache_client.pipeline()
                for pos, arr in zip(miss_positions, tensors, strict=True):
                    pipe.set(
                        _cache_key(model, sequences[pos]),
                        pickle.dumps(arr),
                        ex=cache_ttl_s,
                    )
                await pipe.execute()

            if empty_cache_after:
                try:
                    import torch as _torch

                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    logger.exception("empty_cache failed")

        if any(a is None for a in resolved):
            raise PlmMSAError(
                "Embedding resolution mismatch.",
                code=ErrorCode.INTERNAL,
                http_status=500,
            )
        return resolved, backend.dim

    @app.post("/embed", response_model=EmbedResponse, tags=["embedding"])
    async def embed(req: EmbedRequest) -> EmbedResponse:
        resolved, dim = await _resolve_embeddings(req.model, list(req.sequences))

        def _to_list(x: Any) -> list[list[float]]:
            return x.tolist() if hasattr(x, "tolist") else x

        return EmbedResponse(
            model=req.model,
            dim=dim,
            embeddings=[_to_list(t) for t in resolved],
        )

    @app.post("/embed/bin", tags=["embedding"])
    async def embed_bin(request: Request) -> Response:
        """Binary-response variant of `/embed`.

        Same request body as `/embed` (JSON in — the request is small,
        just a model id + sequence list). Response is the compact
        framing from `plmmsa.align.binary`:
            metadata JSON = {model, dim} ;
            N tensors = one per input sequence, in input order.
        At chunk_size=256, targets ~1000 aa x 1536 dim, this turns a
        ~1.6 GB JSON round-trip into ~1.6 GB of raw f32 bytes that
        decode in <1 s vs. ~30 s for JSON.
        """
        from plmmsa.align import binary as _binary

        body = await request.json()
        model = str(body.get("model", ""))
        sequences = list(body.get("sequences") or [])
        if not model or not sequences:
            raise PlmMSAError(
                "embed/bin: `model` and non-empty `sequences` required.",
                code=ErrorCode.INVALID_FASTA,
                http_status=422,
            )
        resolved, dim = await _resolve_embeddings(model, sequences)
        # Coerce each item to a numpy array once here; `encode_tensors`
        # then tobytes()s directly. CPU torch tensors support .numpy()
        # zero-copy; lists fall back to np.asarray.
        import numpy as _np

        tensors: list[_np.ndarray] = []
        for t in resolved:
            if hasattr(t, "detach"):
                arr = t.detach().cpu().numpy()
            else:
                arr = _np.asarray(t, dtype=_np.float32)
            tensors.append(arr)
        frame = _binary.encode_tensors({"model": model, "dim": dim}, tensors)
        return Response(content=frame, media_type=_binary.CONTENT_TYPE_EMBED)

    @app.post("/embed_by_id", response_model=EmbedByIdResponse, tags=["embedding"])
    async def embed_by_id(req: EmbedByIdRequest) -> EmbedByIdResponse:
        """Serve precomputed per-residue embeddings from the shard store.

        Independent of `/embed`: works even when `settings.models.<name>.enabled`
        is false, as long as the model has a `shard_root` configured. Misses
        are returned for the caller to re-fetch via `/embed` on the
        corresponding sequences. No fallback to model inference here — keeping
        the two paths orthogonal makes them individually testable and makes
        the "cache-only" deployment mode a simple consequence of the design.
        """
        store = shard_stores.get(req.model)
        if store is None:
            raise PlmMSAError(
                f"Model '{req.model}' has no shard store configured.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={
                    "requested": req.model,
                    "shard_models": sorted(shard_stores.keys()),
                },
            )
        found, missing = await asyncio.to_thread(store.fetch, list(req.ids))
        return EmbedByIdResponse(
            model=req.model,
            dim=store.dim,
            found={k: v.tolist() for k, v in found.items()},
            missing=missing,
        )

    @app.post("/embed_by_id/bin", tags=["embedding"])
    async def embed_by_id_bin(request: Request) -> Response:
        """Binary-response variant of /embed_by_id.

        Same request body as /embed_by_id (JSON in, since the request
        is tiny — a list of ids + model name). Response is the compact
        framing from `plmmsa.align.binary`: metadata JSON with
        `{model, dim, found_ids, missing}` + one f32 tensor per id in
        `found_ids` order. At k=1000 this cuts a ~1 GB JSON round-trip
        down to ~600 MB binary that decodes in <1 s.
        """
        from plmmsa.align import binary as _binary

        body = await request.json()
        model = str(body.get("model", ""))
        ids = list(body.get("ids") or [])
        if not model or not ids:
            raise PlmMSAError(
                "embed_by_id/bin: model and non-empty ids required.",
                code=ErrorCode.INVALID_FASTA,
                http_status=422,
            )

        store = shard_stores.get(model)
        if store is None:
            raise PlmMSAError(
                f"Model '{model}' has no shard store configured.",
                code=ErrorCode.UNSUPPORTED_MODEL,
                http_status=400,
                detail={"requested": model, "shard_models": sorted(shard_stores.keys())},
            )

        # Resolve paths via async Redis MGET (falls back to sqlite for
        # any keys Redis doesn't answer). ~300x faster than the old
        # sync sqlite call on /gpfs — was ~13 s per batch of 1500,
        # now ~5 ms. Loads stream immediately after: each resolved
        # `(id, path)` kicks off a torch.load in the default thread
        # pool via asyncio.to_thread, so file I/O overlaps with any
        # remaining resolution work.
        resolved = await store.aresolve_paths(ids)

        async def _one(rid: str, path: Any) -> tuple[str, Any]:
            if path is None:
                return rid, None
            arr = await asyncio.to_thread(store.load_tensor, path)
            return rid, arr

        outcomes = await asyncio.gather(
            *(_one(rid, path) for rid, path in resolved)
        )
        found: dict[str, Any] = {}
        missing: list[str] = []
        for rid, arr in outcomes:
            if arr is None:
                missing.append(rid)
            else:
                found[rid] = arr

        # Keep the ordering deterministic: the key order matches the
        # sequence of tensors so the decoder can zip them directly.
        found_ids = list(found.keys())
        tensors = [found[rid] for rid in found_ids]
        metadata = {
            "model": model,
            "dim": store.dim,
            "found_ids": found_ids,
            "missing": missing,
        }
        frame = _binary.encode_tensors(metadata, tensors)
        return Response(content=frame, media_type=_binary.CONTENT_TYPE_EMBED)

    return app


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _cache_key(model: str, sequence: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(sequence.encode())
    return f"emb:{model}:{h.hexdigest()}"


# Module-level `app` is NOT created here — use `create_app` via uvicorn's
# --factory flag (see `plmmsa/embedding/__main__.py`) so tests and dry-runs do
# not trigger multi-GB model downloads just to import the module.
def _unused(_: Any) -> None: ...
