from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from typing import Any

from redis.asyncio import Redis

from plmmsa.jobs import JobStore, ResultCache
from plmmsa.jobs.models import JobError, JobStatus
from plmmsa.metrics import (
    WORKER_JOBS_PROCESSED,
    WORKER_PIPELINE_DURATION,
    WORKER_QUEUE_DEPTH,
    start_worker_metrics_server,
)
from plmmsa.pipeline import DictTargetFetcher, Orchestrator, OrchestratorConfig, TargetFetcher

logger = logging.getLogger(__name__)


async def _one_iteration(
    store: JobStore,
    orchestrator: Orchestrator,
    result_cache: ResultCache,
) -> bool:
    job = await store.claim_next(timeout_s=1.0)
    if job is None:
        return False
    # Bind the request id the api stamped onto the job (if any) so every
    # downstream HTTP call made by the orchestrator echoes back the
    # same X-Request-ID. Sidecar access logs + api access logs can then
    # be stitched into one trace per MSA submission.
    from plmmsa.request_context import bind_request_id

    request_id = job.request.get("request_id") if isinstance(job.request, dict) else None
    bind_request_id(request_id)
    logger.info("worker: claimed job %s (request_id=%s)", job.id, request_id or "-")

    if job.status == JobStatus.CANCELLED:
        logger.info("worker: job %s was cancelled before we could run it", job.id)
        WORKER_JOBS_PROCESSED.labels(status="cancelled").inc()
        return True

    started_at = time.perf_counter()
    try:
        result = await orchestrator.run(job.request)
    except Exception as exc:
        WORKER_PIPELINE_DURATION.observe(time.perf_counter() - started_at)
        WORKER_JOBS_PROCESSED.labels(status="failed").inc()
        logger.exception("worker: pipeline failed for %s", job.id)
        await store.mark_failed(
            job.id,
            JobError(code="E_INTERNAL", message=str(exc)[:500]),
        )
        return True
    WORKER_PIPELINE_DURATION.observe(time.perf_counter() - started_at)

    refreshed = await store.get(job.id)
    if refreshed is not None and refreshed.status == JobStatus.CANCELLED:
        logger.info("worker: job %s cancelled during execution; discarding result", job.id)
        WORKER_JOBS_PROCESSED.labels(status="cancelled").inc()
        return True

    await store.mark_succeeded(job.id, result)
    # Populate the result cache so the next identical submission can be
    # served immediately by the api without re-queueing. Forced recomputes
    # also overwrite — if the caller asked to regenerate, the fresh answer
    # is what we want cached. Cache writes never block or fail the job;
    # `ResultCache.set` swallows redis errors with a warning log.
    await result_cache.set(job.request, result)
    WORKER_JOBS_PROCESSED.labels(status="succeeded").inc()
    logger.info("worker: job %s succeeded", job.id)
    return True


async def run_forever(
    store: JobStore,
    orchestrator: Orchestrator,
    *,
    result_cache: ResultCache | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    stop_event = stop_event or asyncio.Event()
    cache = result_cache if result_cache is not None else ResultCache(None, ttl_s=0)
    while not stop_event.is_set():
        try:
            await _one_iteration(store, orchestrator, cache)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("worker: unhandled loop error; backing off 2s")
            await asyncio.sleep(2)


async def _sample_queue_depth(store: JobStore, *, stop_event: asyncio.Event) -> None:
    """Sample the queue depth every 5 s for the Prometheus gauge.

    Runs out-of-band so the main drain loop isn't slowed by an extra
    Redis round-trip per iteration. On Redis error we skip the sample
    and keep looping — the last-known value stays in the gauge.
    """
    while not stop_event.is_set():
        try:
            depth = await store.redis.llen(store.queue_key)  # pyright: ignore[reportGeneralTypeIssues]
            WORKER_QUEUE_DEPTH.set(int(depth))
        except Exception:
            logger.warning("worker: queue-depth sample failed", exc_info=True)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=5.0)
        except TimeoutError:
            pass


def _build_fetcher() -> TargetFetcher:
    """Pick a fetcher implementation from the environment.

    Priority:
      1. `PLMMSA_SEQUENCE_REDIS_URL` → RedisTargetFetcher (production path;
         populate with `python -m plmmsa.tools.build_sequence_cache`).
      2. `PLMMSA_TARGET_FASTA` → FastaTargetFetcher (convenient for small
         local corpora; whole FASTA loads into memory).
      3. Fallback → DictTargetFetcher({}). MSAs will contain only the query.
    """
    seq_redis_url = os.environ.get("PLMMSA_SEQUENCE_REDIS_URL")
    if seq_redis_url:
        from redis.asyncio import Redis as _Redis

        from plmmsa.pipeline.fetcher import (
            DEFAULT_SEQ_KEY_FORMAT,
            DEFAULT_TAX_KEY_FORMAT,
            RedisTargetFetcher,
        )

        key_format = os.environ.get("PLMMSA_SEQUENCE_KEY_FORMAT", DEFAULT_SEQ_KEY_FORMAT)
        # `PLMMSA_TAX_KEY_FORMAT` previously fell through to the default
        # (`tax:{id}`) even when host `.env` set `tax:UniRef50_{id}` — the
        # ctor kwarg wasn't wired. Without this, paired-MSA taxonomy joins
        # and unpaired `TaxID=` headers find nothing.
        tax_key_format = os.environ.get("PLMMSA_TAX_KEY_FORMAT", DEFAULT_TAX_KEY_FORMAT)
        logger.info(
            "worker: using Redis sequence cache at %s (seq key %r, tax key %r)",
            seq_redis_url,
            key_format,
            tax_key_format,
        )
        return RedisTargetFetcher(
            _Redis.from_url(seq_redis_url, decode_responses=False),
            key_format=key_format,
            tax_key_format=tax_key_format,
        )

    fasta_path = os.environ.get("PLMMSA_TARGET_FASTA")
    if fasta_path:
        from plmmsa.pipeline.fetcher import FastaTargetFetcher

        logger.info("worker: loading target FASTA from %s", fasta_path)
        return FastaTargetFetcher(fasta_path)

    logger.info("worker: no sequence source configured; MSA will contain only the query")
    return DictTargetFetcher({})


async def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("PLMMSA_WORKER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Prometheus scrape target. Unset → skip; compose ships with :9090
    # bound on the plmmsa_net bridge (no host port mapping).
    metrics_port_env = os.environ.get("PLMMSA_WORKER_METRICS_PORT")
    if metrics_port_env:
        try:
            start_worker_metrics_server(int(metrics_port_env))
        except Exception:
            logger.exception(
                "worker: failed to start metrics server on :%s (continuing without)",
                metrics_port_env,
            )

    cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
    redis = Redis.from_url(cache_url, decode_responses=False)
    store = JobStore(redis)

    # Result cache (cache-emb). Unset → no cache, writes are no-ops and
    # the worker behaves as before. Default TTL 30 days; host `.env` can
    # override via PLMMSA_RESULT_CACHE_TTL_S.
    result_cache_url = os.environ.get("PLMMSA_RESULT_CACHE_URL")
    result_cache_ttl_s = int(os.environ.get("PLMMSA_RESULT_CACHE_TTL_S", 30 * 24 * 60 * 60))
    result_cache_redis: Redis | None = None
    if result_cache_url:
        result_cache_redis = Redis.from_url(result_cache_url, decode_responses=False)
        logger.info(
            "worker: result cache enabled at %s (ttl %ds)",
            result_cache_url,
            result_cache_ttl_s,
        )
    result_cache = ResultCache(result_cache_redis, ttl_s=result_cache_ttl_s)

    from plmmsa.config import get_settings

    settings = get_settings()
    # Comma-separated list of PLM ids whose target embeddings should be
    # looked up in the shard store first via /embed_by_id. Empty → keep
    # today's behavior (always /embed).
    shard_models_env = os.environ.get("PLMMSA_SHARD_MODELS", "")
    shard_models = frozenset(m.strip() for m in shard_models_env.split(",") if m.strip())
    if shard_models:
        logger.info("worker: shard-first target fetch enabled for %s", sorted(shard_models))
    # Collect the set of aligners whose post-score filter is enabled
    # in settings, plus any per-aligner fixed thresholds (OTalign rides
    # the fixed path because its transport-mass score is on a different
    # scale than the upstream (0.2*L, 8.0) dot-product calibration).
    aligner_ids = ("plmalign", "plm_blast", "otalign")
    filter_enabled_aligners = frozenset(
        aid
        for aid in aligner_ids
        if getattr(getattr(settings.aligners, aid, None), "filter_enabled", False)
    )
    filter_thresholds: dict[str, float] = {}
    for aid in aligner_ids:
        entry = getattr(settings.aligners, aid, None)
        threshold = getattr(entry, "filter_threshold", None)
        if threshold is not None:
            filter_thresholds[aid] = float(threshold)
    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            embedding_url=os.environ.get("EMBEDDING_URL", "http://embedding:8081"),
            vdb_url=os.environ.get("VDB_URL", "http://vdb:8082"),
            align_url=os.environ.get("ALIGN_URL", "http://align:8083"),
            default_k=settings.queue.default_k,
            embed_chunk_size=settings.queue.embed_chunk_size,
            paired_k_multiplier=settings.queue.paired_k_multiplier,
            shard_models=shard_models,
            filter_enabled_aligners=filter_enabled_aligners,
            filter_thresholds=filter_thresholds,
            # score_model is resolved at the API edge per-aligner and
            # stamped into the job payload. cfg.score_model stays empty as
            # a no-op safety belt for non-API callers (bench scripts).
        ),
        fetcher=_build_fetcher(),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(*_: Any) -> None:
        logger.info("worker: shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _shutdown)

    depth_task = asyncio.create_task(_sample_queue_depth(store, stop_event=stop_event))
    try:
        await run_forever(
            store,
            orchestrator,
            result_cache=result_cache,
            stop_event=stop_event,
        )
    finally:
        depth_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await depth_task
        await redis.aclose()
        if result_cache_redis is not None:
            try:
                await result_cache_redis.aclose()
            except Exception:
                logger.exception("worker: result_cache redis close failed")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
