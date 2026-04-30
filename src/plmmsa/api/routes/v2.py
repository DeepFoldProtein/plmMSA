from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from plmmsa import __version__
from plmmsa.api.auth import require_admin_token
from plmmsa.api.middleware import audit_event
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.jobs import JobStore, ResultCache
from plmmsa.jobs.models import Job, JobStatus

_IDEMPOTENCY_TTL_S = 600  # 10 min — long enough to de-dupe retries, short
# enough that tuning a submission isn't blocked.

# Result-cache TTL default: 30 days. Stored entries are also governed by
# cache-emb's `allkeys-lru` + `CACHE_EMB_MAXMEMORY` eviction, so this is a
# ceiling, not a floor.
_RESULT_CACHE_TTL_S_DEFAULT = 30 * 24 * 60 * 60

router = APIRouter(tags=["v2"])

_job_store: JobStore | None = None
_result_cache: ResultCache | None = None

# Module-level orchestrator handle. Tests override via
# `plmmsa.api.routes.v2._templates_orchestrator = <fake>`. The
# `Any` typing avoids a circular module-level import — the real type
# is `plmmsa.templates.TemplatesRealignOrchestrator`.
_templates_orchestrator: Any | None = None


async def _get_job_store() -> JobStore:
    """Lazy singleton — production uses CACHE_URL; tests override via
    `plmmsa.api.routes.v2._job_store = <fake>`."""
    global _job_store
    if _job_store is None:
        cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
        _job_store = JobStore(Redis.from_url(cache_url, decode_responses=False))
    return _job_store


async def _get_result_cache() -> ResultCache:
    """Lazy singleton for the MSA result cache on ``cache-emb``.

    Production sets ``PLMMSA_RESULT_CACHE_URL`` to ``redis://cache-emb:6379``.
    Leaving it unset disables the cache (get/set become no-ops) so the
    test harness and non-docker dev loops don't need a second Redis.
    Tests can override via ``plmmsa.api.routes.v2._result_cache = <fake>``.
    """
    global _result_cache
    if _result_cache is None:
        url = os.environ.get("PLMMSA_RESULT_CACHE_URL")
        ttl_s = int(os.environ.get("PLMMSA_RESULT_CACHE_TTL_S", _RESULT_CACHE_TTL_S_DEFAULT))
        client = Redis.from_url(url, decode_responses=False) if url else None
        _result_cache = ResultCache(client, ttl_s=ttl_s)
    return _result_cache


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
    # Prefer `models` (list). `model` is kept for single-PLM callers and
    # older clients — when both are unset, the API resolves to every enabled
    # PLM that has a live VDB collection (today: ankh_cl + esm1b) and the
    # orchestrator runs them in parallel, unioning hits by target id.
    models: list[str] | None = Field(
        None,
        description=(
            "PLM backend ids to run in parallel; hits are unioned by target "
            "id. Defaults to every enabled model with a VDB collection."
        ),
    )
    model: str | None = Field(
        None,
        description="Legacy single-PLM escape hatch. Prefer `models`.",
    )
    output_format: str = Field("a3m", description="Wire format: a3m | stockholm | ...")
    paired: bool = Field(False, description="Produce paired MSA across chains.")
    # Orchestrator knobs (optional; the orchestrator supplies defaults).
    query_ids: list[str] | None = Field(
        None,
        description=(
            "Per-chain FASTA header labels, one per entry in `sequences`. "
            "When omitted, labels default to ['101', '102', ...]."
        ),
    )
    collection: str | None = Field(
        None,
        description=(
            "Single-PLM VDB collection override. Ignored when `models` has "
            "more than one entry — use `collections` for per-model pinning."
        ),
    )
    collections: dict[str, str] | None = Field(
        None,
        description="Map of model id → VDB collection id (per-model override).",
    )
    k: int | None = Field(
        None, ge=1, le=10_000, description="Number of FAISS neighbors to fetch per model."
    )
    aligner: str | None = Field(None, description="Aligner id (default: plm_blast).")
    mode: str | None = Field(
        None,
        description=(
            "Alignment mode. `local` / `global` are supported by every "
            "aligner. `glocal`, `q2t`, `t2q` are OTalign-only "
            "(PLMAlign / pLM-BLAST reject them with 400)."
        ),
    )
    score_model: str | None = Field(
        None,
        description=(
            "PLM used to build the score matrix. Defaults to the server's "
            "`queue.default_score_model` (upstream PLMAlign uses ProtT5). "
            "Pass an empty string to skip cross-PLM scoring and align with "
            "the same PLM that searched the VDB."
        ),
    )
    options: dict[str, Any] | None = Field(
        None, description="Pass-through aligner kwargs (gap_open, gap_extend, ...)."
    )
    filter_by_score: bool = Field(
        True,
        description=(
            "Apply the post-alignment score-threshold filter. The cutoff "
            "is aligner-specific: PLMAlign / pLM-BLAST use upstream "
            "Algorithm 1 step 5 (`min(0.2 * top_hit_score, 8.0)` — see "
            "PLMAlign's alignment_to_a3m.py), OTalign uses its calibrated "
            "transport-mass floor (`[aligners.otalign].filter_threshold`, "
            "0.25 by default). Set false to return every aligned hit."
        ),
    )
    force_recompute: bool = Field(
        False,
        description=(
            "Bypass the result cache and always run the full pipeline. "
            "Useful for reproducing historical MSAs or when retrying "
            "after a pipeline change. On success the fresh result still "
            "overwrites the cache entry."
        ),
    )


class SubmitResponse(BaseModel):
    job_id: str
    status: str
    status_url: str


_ALLOWED_RESIDUE_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZJUO")
_KNOWN_MODELS = ("ankh_cl", "ankh_large", "esm1b", "prott5")


def _default_models_with_vdb(settings: Any) -> list[str]:
    """Every enabled PLM that has an enabled VDB collection to search.

    Today this is `ankh_cl` (ankh_uniref50) + `esm1b` (esm1b_uniref50).
    Ankh-Large and ProtT5 are enabled for `/v2/embed` but lack dedicated
    UniRef50 FAISS indexes, so they can't contribute hits and are excluded
    from the aggregate default.
    """
    backends_with_vdb = {c.model_backend for c in settings.vdb.collections.values() if c.enabled}
    return sorted(
        name
        for name in _KNOWN_MODELS
        if getattr(settings.models, name).enabled and name in backends_with_vdb
    )


def _resolve_models(req: SubmitRequest, settings: Any) -> list[str]:
    """Pick the effective model list for the job.

    Precedence: explicit `models` > legacy `model` > aggregate default.
    Duplicates are collapsed while preserving caller order.
    """
    if req.models:
        seen: dict[str, None] = {}
        for m in req.models:
            seen.setdefault(m, None)
        return list(seen.keys())
    if req.model:
        return [req.model]
    return _default_models_with_vdb(settings)


def _resolve_score_model(req: SubmitRequest, settings: Any) -> str:
    """Pick the effective score-model for the job.

    Precedence (highest wins):
      1. `req.score_model` (including `""` which means "opt out of cross-PLM")
      2. `settings.aligners.<resolved_aligner>.score_model`

    Aligner defaults encode upstream convention:
      - plmalign + plm_blast → ProtT5
      - otalign              → Ankh-Large

    Returns `""` to skip cross-PLM and fall back to the per-model aggregate
    path (each retrieval PLM scores its own hits).
    """
    if req.score_model is not None:
        return req.score_model.strip()
    # Look up the aligner that will actually run: explicit request value >
    # orchestrator default. We keep these in sync by reading the dataclass
    # default via a small helper on the orchestrator module.
    from plmmsa.pipeline.orchestrator import default_aligner_id

    aligner_id = req.aligner or default_aligner_id()
    entry = getattr(settings.aligners, aligner_id, None)
    return str(getattr(entry, "score_model", "") or "").strip()


def _resolve_collections_for_models(
    resolved_models: list[str],
    settings: Any,
) -> dict[str, str]:
    """Look up each retrieval model's VDB collection by matching
    `vdb.collections.<name>.model_backend`. The orchestrator's old
    `f"{model}_uniref50"` default was fragile — it happened to match
    `esm1b_uniref50` but not `ankh_uniref50` (Ankh-CL's VDB drops the
    `_cl` suffix by convention).
    """
    by_backend: dict[str, str] = {}
    for coll_id, coll_cfg in settings.vdb.collections.items():
        if getattr(coll_cfg, "enabled", True):
            by_backend[coll_cfg.model_backend] = coll_id
    return {m: by_backend[m] for m in resolved_models if m in by_backend}


def _resolve_query_ids(req: SubmitRequest) -> list[str]:
    """Normalize `query_ids` into one canonical list, one entry per chain.

    When the caller omits it, default to ``["101", "102", ...]`` — stable
    integer labels per-chain so the emitted A3M always has a non-empty
    header even on anonymous submissions.
    """
    if req.query_ids is not None:
        return list(req.query_ids)
    return [str(101 + i) for i in range(len(req.sequences))]


def _validate_submit(req: SubmitRequest, settings: Any) -> tuple[list[str], str]:
    """Edge validation enforced before anything touches the queue.

    Returns `(resolved_models, resolved_score_model)` so the caller can
    stamp them onto the job record. Worker then runs exactly what the
    API promised — no drift from settings changes that happen between
    submit and claim.
    """
    chains = req.sequences
    if req.paired and len(chains) > settings.limits.max_chains_paired:
        raise PlmMSAError(
            f"Paired MSA supports up to {settings.limits.max_chains_paired} chains.",
            code=ErrorCode.TOO_MANY_CHAINS,
            http_status=400,
            detail={"chains": len(chains), "max": settings.limits.max_chains_paired},
        )

    # query_ids (when set) must have one label per chain.
    if req.query_ids is not None and len(req.query_ids) != len(chains):
        raise PlmMSAError(
            f"query_ids has {len(req.query_ids)} entries but sequences "
            f"has {len(chains)}; they must match.",
            code=ErrorCode.INVALID_FASTA,
            http_status=400,
            detail={
                "query_ids": len(req.query_ids),
                "sequences": len(chains),
            },
        )
    max_len = settings.limits.max_residues_per_chain
    for idx, seq in enumerate(chains):
        stripped = seq.strip()
        if not stripped:
            raise PlmMSAError(
                f"sequences[{idx}] is empty.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
                detail={"chain": idx},
            )
        if len(stripped) > max_len:
            raise PlmMSAError(
                f"sequences[{idx}] is {len(stripped)} residues; max is {max_len}.",
                code=ErrorCode.SEQ_TOO_LONG,
                http_status=400,
                detail={"chain": idx, "length": len(stripped), "max": max_len},
            )
        bad = sorted({c for c in stripped.upper() if c not in _ALLOWED_RESIDUE_CHARS})
        if bad:
            raise PlmMSAError(
                f"sequences[{idx}] has non-amino-acid characters: {''.join(bad)}",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
                detail={"chain": idx, "invalid_chars": bad},
            )

    enabled_models = {name for name in _KNOWN_MODELS if getattr(settings.models, name).enabled}
    resolved = _resolve_models(req, settings)
    if not resolved:
        raise PlmMSAError(
            "No PLM with a VDB collection is enabled — nothing to run.",
            code=ErrorCode.UNSUPPORTED_MODEL,
            http_status=400,
            detail={"available": sorted(enabled_models)},
        )
    bad_models = [m for m in resolved if m not in enabled_models]
    if bad_models:
        raise PlmMSAError(
            f"Unknown / disabled models: {bad_models}",
            code=ErrorCode.UNSUPPORTED_MODEL,
            http_status=400,
            detail={"requested": bad_models, "available": sorted(enabled_models)},
        )

    # score_model: empty string = opt out of cross-PLM scoring (fine).
    # Any non-empty value must be a known + enabled PLM — the embedding
    # service would 400 later anyway, but failing fast keeps the queue
    # clean and gives the client a stable error code at submit time.
    resolved_score = _resolve_score_model(req, settings)
    if resolved_score and resolved_score not in enabled_models:
        raise PlmMSAError(
            f"score_model {resolved_score!r} is not a known / enabled PLM.",
            code=ErrorCode.UNSUPPORTED_MODEL,
            http_status=400,
            detail={
                "requested": resolved_score,
                "available": sorted(enabled_models),
            },
        )
    return resolved, resolved_score


async def _enforce_backpressure(store: JobStore, settings: Any) -> None:
    """Soft + hard queue caps. Soft returns 503 with Retry-After so clients
    back off politely; hard is identical today but kept distinct so we can
    tune later without changing the client contract."""
    depth = await store.redis.llen(store.queue_key)  # pyright: ignore[reportGeneralTypeIssues]
    if depth >= settings.queue.max_queue_depth:
        raise PlmMSAError(
            "Job queue is at hard capacity; try again later.",
            code=ErrorCode.QUEUE_FULL,
            http_status=503,
            detail={"depth": depth, "max": settings.queue.max_queue_depth},
            headers={"Retry-After": "30"},
        )
    if depth >= settings.queue.backpressure_threshold:
        raise PlmMSAError(
            "Job queue is under backpressure; try again shortly.",
            code=ErrorCode.QUEUE_FULL,
            http_status=503,
            detail={
                "depth": depth,
                "threshold": settings.queue.backpressure_threshold,
            },
            headers={"Retry-After": "5"},
        )


_IDEMPOTENCY_EXCLUDED_FIELDS = frozenset({"request_id"})


def _idempotency_key(scope_id: str | None, payload: dict[str, Any]) -> str:
    """Hash the (scope, canonical payload) pair.

    Sorting keys makes the hash invariant to JSON key ordering. Scope is the
    caller's token id when authenticated, otherwise the client IP, so two
    clients with the same payload don't collide (and so one client's retries
    de-dup). For truly public submission, IP-scoping is a best-effort: a NAT
    fleet may de-dup across members, which is fine — they'd get the same
    answer from the pipeline anyway.

    ``request_id`` is excluded from the canonical form — two submits
    with the same scientific parameters should de-dup regardless of
    which edge request stamped them.
    """
    canonical_payload = {k: v for k, v in payload.items() if k not in _IDEMPOTENCY_EXCLUDED_FIELDS}
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256()
    h.update((scope_id or "anon").encode())
    h.update(b"\x00")
    h.update(canonical)
    return f"idem:msa:{h.hexdigest()}"


@router.post("/msa", status_code=202, response_model=SubmitResponse)
async def submit_msa(req: SubmitRequest, request: Request) -> SubmitResponse:
    settings = get_settings()
    resolved_models, resolved_score_model = _validate_submit(req, settings)
    store = await _get_job_store()
    token_id = getattr(request.state, "token_id", None)
    request_id = getattr(request.state, "request_id", None)
    client_ip = getattr(request.state, "client_ip", None)

    # Stamp the resolved model list + score_model onto the job payload so
    # the worker runs exactly what the API promised, and so the record on
    # disk shows what actually happened (not the raw caller request which
    # may have omitted either field). Drop the legacy single-model field
    # to avoid ambiguity — `models` is the source of truth from here on.
    payload = req.model_dump()
    payload["models"] = resolved_models
    payload["score_model"] = resolved_score_model
    # Thread the incoming request id onto the job so the worker can
    # forward it as X-Request-ID on every downstream call. Excluded
    # from the result-cache canonicalization — see _CACHED_FIELDS.
    if request_id:
        payload["request_id"] = request_id
    # Per-chain FASTA header labels — default to ['101', '102', ...]
    # when the caller doesn't pin them, so the emitted A3M always has
    # deterministic headers.
    payload["query_ids"] = _resolve_query_ids(req)
    payload.pop("query_id", None)
    # Resolve VDB collection per retrieval model from settings (via the
    # `model_backend` reverse map) so the orchestrator doesn't fall back
    # to the fragile `f"{model}_uniref50"` convention. Caller-supplied
    # `collections` still wins per entry.
    resolved_collections = _resolve_collections_for_models(resolved_models, settings)
    caller_collections = req.collections or {}
    merged_collections = {**resolved_collections, **dict(caller_collections)}
    payload["collections"] = merged_collections
    payload.pop("model", None)

    # Idempotency: same (scope, canonical payload) within TTL returns the
    # same job id. Prevents double-submission on client retries without
    # requiring the client to send its own idempotency-key header. When
    # submissions are anonymous, scope falls back to the client IP.
    idem_key = _idempotency_key(token_id or client_ip, payload)
    prior = await store.redis.get(idem_key)
    if prior is not None:
        prior_id = prior.decode("utf-8") if isinstance(prior, bytes) else prior
        prior_job = await store.get(prior_id)
        # Only deduplicate against a prior that still has a chance of
        # succeeding. If the prior failed or was cancelled, the caller
        # almost certainly wants a fresh attempt (the failure may have
        # been an infra blip, not a deterministic property of the
        # request). Skip dedup in that case and fall through to
        # creating a new job.
        if prior_job is not None and prior_job.status.value not in ("failed", "cancelled"):
            audit_event(
                "msa.submit.idempotent",
                token_id=token_id,
                request_id=request_id,
                client_ip=client_ip,
                job_id=prior_id,
            )
            return SubmitResponse(
                job_id=prior_id,
                status=prior_job.status.value,
                status_url=f"/v2/msa/{prior_id}",
            )

    # Result-cache hit → synthesize a succeeded job record so clients
    # still poll via GET /v2/msa/{id} without needing to know about the
    # cache. The payload stamp on the synthetic job carries the original
    # submit (so regeneration preserves the original parameters) and the
    # result's `stats` gets a `cache_hit: true` marker so observability
    # downstream can distinguish served-from-cache from fresh compute.
    # Clients can force a bypass with `force_recompute=true`.
    if not req.force_recompute:
        cache = await _get_result_cache()
        cached = await cache.get(payload)
        if cached is not None:
            synthesized = Job(
                id=str(uuid.uuid4()),
                status=JobStatus.SUCCEEDED,
                request=payload,
                created_at=time.time(),
            )
            synthesized.started_at = synthesized.created_at
            synthesized.finished_at = synthesized.created_at
            stats = dict(cached.stats)
            stats["cache_hit"] = True
            synthesized.result = cached.model_copy(update={"stats": stats})
            await store.insert_terminal(synthesized)
            await store.redis.set(idem_key, synthesized.id, ex=_IDEMPOTENCY_TTL_S)  # pyright: ignore[reportGeneralTypeIssues]
            audit_event(
                "msa.submit.cache_hit",
                token_id=token_id,
                request_id=request_id,
                client_ip=client_ip,
                job_id=synthesized.id,
            )
            return SubmitResponse(
                job_id=synthesized.id,
                status=synthesized.status.value,
                status_url=f"/v2/msa/{synthesized.id}",
            )

    await _enforce_backpressure(store, settings)
    job = await store.create(payload)
    await store.redis.set(idem_key, job.id, ex=_IDEMPOTENCY_TTL_S)  # pyright: ignore[reportGeneralTypeIssues]
    audit_event(
        "msa.submit",
        token_id=token_id,
        request_id=request_id,
        client_ip=client_ip,
        job_id=job.id,
        models=resolved_models,
        score_model=resolved_score_model or None,
        paired=req.paired,
        chain_count=len(req.sequences),
    )
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
async def cancel_msa(request: Request, job_id: str = Path(..., min_length=1)) -> Response:
    store = await _get_job_store()
    job = await store.cancel(job_id)
    if job is None:
        raise PlmMSAError(
            f"Job {job_id} not found.",
            code=ErrorCode.JOB_NOT_FOUND,
            http_status=404,
            detail={"job_id": job_id},
        )
    audit_event(
        "msa.cancel",
        token_id=getattr(request.state, "token_id", None),
        request_id=getattr(request.state, "request_id", None),
        client_ip=getattr(request.state, "client_ip", None),
        job_id=job_id,
    )
    return Response(status_code=204)


async def _forward(
    target: str,
    path: str,
    payload: dict,
    *,
    timeout: float = 120.0,
    service: str,
    request: Request | None = None,
) -> JSONResponse:
    """Common httpx passthrough used by /v2/embed, /v2/search, /v2/align.

    `target` is the internal URL (e.g. `http://embedding:8081`); it's never
    surfaced to clients. `service` is the public-facing label ("embedding",
    "vdb", "align") used in error messages so operators can tell where the
    failure came from without leaking the internal hostname. When a `request`
    is provided, the X-Request-ID set by `RequestContextMiddleware` is
    forwarded upstream so sidecar logs can be correlated.
    """
    import logging as _log

    from plmmsa.request_context import httpx_headers_with_request_id

    headers: dict[str, str] = {}
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if rid:
            headers["X-Request-ID"] = rid
    # Fallback to the ContextVar (always set by RequestContextMiddleware)
    # — covers callers that don't pass `request=` explicitly.
    headers = httpx_headers_with_request_id(headers)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.post(f"{target}{path}", json=payload, headers=headers)
    except httpx.HTTPError as exc:
        _log.getLogger("plmmsa.forward").warning(
            "upstream unreachable",
            extra={"service": service, "target": target, "cause": str(exc)[:200]},
        )
        raise PlmMSAError(
            f"Upstream {service} service unreachable.",
            code=ErrorCode.INTERNAL,
            http_status=502,
            detail={"service": service},
        ) from exc

    try:
        body = upstream.json()
    except ValueError as exc:
        _log.getLogger("plmmsa.forward").warning(
            "upstream non-json",
            extra={"service": service, "status": upstream.status_code},
        )
        raise PlmMSAError(
            f"Upstream {service} service returned non-JSON.",
            code=ErrorCode.INTERNAL,
            http_status=502,
            detail={"service": service, "status": upstream.status_code},
        ) from exc

    return JSONResponse(status_code=upstream.status_code, content=body)


class EmbedRequest(BaseModel):
    model: str = Field("ankh_cl", description="PLM backend id.")
    sequences: list[str] = Field(..., min_length=1)


@router.post("/embed", dependencies=[Depends(require_admin_token)])
async def embed(req: EmbedRequest, request: Request) -> JSONResponse:
    """Forward raw per-residue embeddings from the embedding service.

    Bearer-token gated (see `plmmsa.api.auth.require_admin_token`).
    """
    target = os.environ.get("EMBEDDING_URL", "http://embedding:8081")
    return await _forward(
        target,
        "/embed",
        req.model_dump(),
        service="embedding",
        request=request,
    )


class SearchRequest(BaseModel):
    collection: str = Field(..., description="VDB collection id.")
    vectors: list[list[float]] = Field(..., min_length=1)
    k: int = Field(100, ge=1, le=10_000)
    nprobe: int | None = Field(None, ge=1, le=10_000)


@router.post("/search", dependencies=[Depends(require_admin_token)])
async def search(req: SearchRequest, request: Request) -> JSONResponse:
    """Forward a vector search to the VDB service.

    Same auth model as `/v2/embed`: bearer token against a minted API token
    (or the bootstrap `ADMIN_TOKEN`).
    """
    target = os.environ.get("VDB_URL", "http://vdb:8082")
    return await _forward(
        target,
        "/search",
        req.model_dump(),
        service="vdb",
        request=request,
    )


class AlignRequest(BaseModel):
    aligner: str = Field("plmalign", description="Aligner id.")
    mode: str = Field("local", description="'local' or 'global'.")
    query_embedding: list[list[float]] = Field(..., min_length=1)
    target_embeddings: list[list[list[float]]] = Field(..., min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/align", dependencies=[Depends(require_admin_token)])
async def align(req: AlignRequest, request: Request) -> JSONResponse:
    """Forward a pairwise alignment to the align service.

    Same auth model as `/v2/embed` and `/v2/search`. Aligner-specific
    tunables (gap_open, gap_extend, normalize, ...) ride in `options`.
    """
    target = os.environ.get("ALIGN_URL", "http://align:8083")
    return await _forward(
        target,
        "/align",
        req.model_dump(),
        service="align",
        request=request,
    )


# ---------------------------------------------------------------------------
# /v2/templates/realign — re-align an existing hmmsearch a3m
# ---------------------------------------------------------------------------


class TemplatesRealignBody(BaseModel):
    """Request body for `/v2/templates/realign`. See
    `PLAN_TEMPLATES_REALIGN.md` §3 for the contract.
    """

    query_id: str = Field(
        "query",
        min_length=1,
        max_length=128,
        description="A3M label for the query record at the top of the output.",
    )
    query_sequence: str = Field(
        ...,
        min_length=1,
        description=(
            "Query residues. Normalized server-side: uppercase, "
            "whitespace-stripped, gap-stripped. Must match the "
            "match-state count of every record in `a3m`."
        ),
    )
    a3m: str = Field(
        ...,
        min_length=1,
        description="hmmsearch-style A3M body (one or more records).",
    )
    model: str | None = Field(
        None,
        description="PLM backend id; default `ankh_large`.",
    )
    mode: str | None = Field(
        None,
        description="OTalign DP mode; default `glocal`.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra OTalign tunables passed straight through.",
    )
    sort_by_score: bool = Field(
        False,
        description=(
            "When true, output records are emitted in OTalign-score-"
            "descending order (best hit first). Default false preserves "
            "input order — useful for diffing against the original "
            "hmmsearch a3m row-by-row."
        ),
    )


class TemplatesRealignResponseBody(BaseModel):
    format: str
    payload: str
    stats: dict[str, Any]


def _build_templates_orchestrator() -> Any:
    """Construct the production orchestrator from env vars.

    Lives in its own function so tests can monkeypatch it (or just
    override `_templates_orchestrator` directly to skip building).
    """
    from plmmsa.templates import (
        HttpTransport,
        TemplatesRealignConfig,
        TemplatesRealignOrchestrator,
    )

    embedding_url = os.environ.get("EMBEDDING_URL", "http://embedding:8081")
    align_url = os.environ.get("ALIGN_URL", "http://align:8083")
    timeout_s = float(os.environ.get("PLMMSA_TEMPLATES_TIMEOUT_S", "900"))

    transport = HttpTransport(
        embedding_url=embedding_url,
        align_url=align_url,
        timeout_s=timeout_s,
    )
    return TemplatesRealignOrchestrator(
        config=TemplatesRealignConfig(),
        transport=transport,
    )


def _get_templates_orchestrator() -> Any:
    global _templates_orchestrator
    if _templates_orchestrator is None:
        _templates_orchestrator = _build_templates_orchestrator()
    return _templates_orchestrator


@router.post(
    "/templates/realign",
    response_model=TemplatesRealignResponseBody,
    dependencies=[Depends(require_admin_token)],
)
async def templates_realign(
    body: TemplatesRealignBody,
    request: Request,
) -> TemplatesRealignResponseBody:
    """Re-align an existing hmmsearch-style A3M against the query under
    OTalign / Ankh-Large / glocal.

    Sync endpoint — runs the orchestrator inline and returns the result
    JSON. Async/job lifecycle (PLAN §3 submit-then-poll) lands in a
    follow-up; for now, callers with very large inputs should expect
    the request to take minutes.
    """
    from plmmsa.templates import TemplatesRealignRequest

    orch = _get_templates_orchestrator()
    result = await orch.run(
        TemplatesRealignRequest(
            query_id=body.query_id,
            query_sequence=body.query_sequence,
            a3m=body.a3m,
            model=body.model,
            mode=body.mode,
            options=dict(body.options),
            sort_by_score=body.sort_by_score,
        )
    )
    return TemplatesRealignResponseBody(
        format=result.format,
        payload=result.payload,
        stats=result.stats,
    )
