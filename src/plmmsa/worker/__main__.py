from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from typing import Any

from redis.asyncio import Redis

from plmmsa.jobs import JobStore
from plmmsa.jobs.models import JobError, JobStatus
from plmmsa.pipeline import DictTargetFetcher, Orchestrator, OrchestratorConfig, TargetFetcher

logger = logging.getLogger(__name__)


async def _one_iteration(store: JobStore, orchestrator: Orchestrator) -> bool:
    job = await store.claim_next(timeout_s=1.0)
    if job is None:
        return False
    logger.info("worker: claimed job %s", job.id)

    if job.status == JobStatus.CANCELLED:
        logger.info("worker: job %s was cancelled before we could run it", job.id)
        return True

    try:
        result = await orchestrator.run(job.request)
    except Exception as exc:
        logger.exception("worker: pipeline failed for %s", job.id)
        await store.mark_failed(
            job.id,
            JobError(code="E_INTERNAL", message=str(exc)[:500]),
        )
        return True

    refreshed = await store.get(job.id)
    if refreshed is not None and refreshed.status == JobStatus.CANCELLED:
        logger.info("worker: job %s cancelled during execution; discarding result", job.id)
        return True

    await store.mark_succeeded(job.id, result)
    logger.info("worker: job %s succeeded", job.id)
    return True


async def run_forever(
    store: JobStore,
    orchestrator: Orchestrator,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    stop_event = stop_event or asyncio.Event()
    while not stop_event.is_set():
        try:
            await _one_iteration(store, orchestrator)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("worker: unhandled loop error; backing off 2s")
            await asyncio.sleep(2)


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

        from plmmsa.pipeline.fetcher import DEFAULT_SEQ_KEY_FORMAT, RedisTargetFetcher

        key_format = os.environ.get("PLMMSA_SEQUENCE_KEY_FORMAT", DEFAULT_SEQ_KEY_FORMAT)
        logger.info(
            "worker: using Redis sequence cache at %s (key format %r)",
            seq_redis_url,
            key_format,
        )
        return RedisTargetFetcher(
            _Redis.from_url(seq_redis_url, decode_responses=False),
            key_format=key_format,
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

    cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
    redis = Redis.from_url(cache_url, decode_responses=False)
    store = JobStore(redis)

    orchestrator = Orchestrator(
        config=OrchestratorConfig(
            embedding_url=os.environ.get("EMBEDDING_URL", "http://embedding:8081"),
            vdb_url=os.environ.get("VDB_URL", "http://vdb:8082"),
            align_url=os.environ.get("ALIGN_URL", "http://align:8083"),
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

    try:
        await run_forever(store, orchestrator, stop_event=stop_event)
    finally:
        await redis.aclose()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
