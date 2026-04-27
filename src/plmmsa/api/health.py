"""API health endpoints — bare liveness and aggregated readiness.

Two endpoints live here:

- ``GET /healthz`` — bare liveness probe. Returns 200 when the api
  process is up, regardless of downstream state. Used by docker-compose
  healthchecks and the Cloudflare edge probe, where a slow aggregate
  would cause harmful flapping.

- ``GET /health`` — aggregated readiness. Fans out in parallel to
  ``embedding``, ``vdb``, ``align`` and pings each Redis role
  (``cache-ops``, ``cache-seq``, ``cache-emb``). ColabFold notebooks
  and any other external client use this as a warmup gate before
  submitting a job. Cached for ~1.5 s so a burst of polls doesn't fan
  six probes per request.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel
from redis.asyncio import Redis

router = APIRouter()
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    service: str


class DownstreamStatus(BaseModel):
    status: str  # "ok" | "down"
    detail: dict[str, Any] | None = None


class AggregatedHealthResponse(BaseModel):
    status: str  # "ok" | "down"
    service: str
    downstream: dict[str, DownstreamStatus]


@router.get("/healthz", response_model=HealthResponse, tags=["system"])
async def healthz() -> HealthResponse:
    """Bare liveness probe — no downstream fan-out.

    Cheap, always returns 200 when the api process is alive. Used by
    docker-compose healthchecks and the Cloudflare edge probe.
    """
    return HealthResponse(status="ok", service="api")


# ---------------------------------------------------------------------------
# Aggregated /health — TTL-cached so polling clients don't amplify load.
# 1.5 s is short enough that a warmup probe sees fresh state but long
# enough to de-duplicate a tight poll loop.
# ---------------------------------------------------------------------------

_AGGREGATE_TTL_S = 1.5
_PROBE_TIMEOUT_S = 2.0
_AGGREGATE_DEADLINE_S = 3.0

_aggregate_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_aggregate_lock = asyncio.Lock()


async def _probe_http(
    client: httpx.AsyncClient, service: str, url: str
) -> tuple[str, DownstreamStatus]:
    try:
        resp = await client.get(url, timeout=_PROBE_TIMEOUT_S)
        if resp.status_code == 200:
            try:
                body = resp.json()
            except Exception:
                body = {"note": "non-json health body"}
            return service, DownstreamStatus(status="ok", detail=body)
        return service, DownstreamStatus(
            status="down",
            detail={"http_status": resp.status_code, "body": resp.text[:200]},
        )
    except Exception as exc:
        return service, DownstreamStatus(status="down", detail={"error": str(exc)[:200]})


async def _probe_redis(service: str, url: str) -> tuple[str, DownstreamStatus]:
    client = Redis.from_url(url, decode_responses=False)
    try:
        await asyncio.wait_for(client.ping(), timeout=_PROBE_TIMEOUT_S)
        return service, DownstreamStatus(status="ok")
    except Exception as exc:
        return service, DownstreamStatus(status="down", detail={"error": str(exc)[:200]})
    finally:
        try:
            await client.aclose()
        except Exception:
            logger.debug("health: redis aclose failed for %s", service, exc_info=True)


@router.get("/health", response_model=AggregatedHealthResponse, tags=["system"])
async def health() -> AggregatedHealthResponse:
    """Aggregated readiness — fans out to every downstream.

    Use as a warmup gate: a client polls ``/health`` until ``status == "ok"``
    before submitting. The overall status is ``ok`` only when every
    downstream probe returns ok; any ``down`` demotes the aggregate.

    Cached for ~1.5 s via an ``asyncio.Lock``-guarded TTL so a burst of
    polls doesn't fan six HTTP + Redis probes per request.
    """
    now = time.monotonic()
    cached = _aggregate_cache.get("payload")
    if cached is not None and now - _aggregate_cache["ts"] < _AGGREGATE_TTL_S:
        return cached

    async with _aggregate_lock:
        # Double-checked: another coroutine may have refreshed while we waited.
        now = time.monotonic()
        cached = _aggregate_cache.get("payload")
        if cached is not None and now - _aggregate_cache["ts"] < _AGGREGATE_TTL_S:
            return cached

        embedding_url = os.environ.get("EMBEDDING_URL", "http://embedding:8081")
        vdb_url = os.environ.get("VDB_URL", "http://vdb:8082")
        align_url = os.environ.get("ALIGN_URL", "http://align:8083")
        cache_ops_url = os.environ.get("CACHE_URL", "redis://cache-ops:6379")
        cache_seq_url = os.environ.get("PLMMSA_SEQUENCE_REDIS_URL", "redis://cache-seq:6379")
        # cache-emb is deployed but not currently read from by any service
        # (per the embedding-cache removal). Ping it anyway so the aggregate
        # surfaces outages early; the forthcoming result cache will depend
        # on it being reachable.
        cache_emb_url = os.environ.get("PLMMSA_RESULT_CACHE_URL", "redis://cache-emb:6379")

        downstream: dict[str, DownstreamStatus] = {}
        try:
            async with httpx.AsyncClient() as client:
                probes = await asyncio.wait_for(
                    asyncio.gather(
                        _probe_http(client, "embedding", f"{embedding_url}/health"),
                        _probe_http(client, "vdb", f"{vdb_url}/health"),
                        _probe_http(client, "align", f"{align_url}/health"),
                        _probe_redis("cache-ops", cache_ops_url),
                        _probe_redis("cache-seq", cache_seq_url),
                        _probe_redis("cache-emb", cache_emb_url),
                    ),
                    timeout=_AGGREGATE_DEADLINE_S,
                )
            for name, status in probes:
                downstream[name] = status
        except TimeoutError:
            logger.warning("health: aggregate deadline exceeded; marking unresolved probes down")
            for name in ("embedding", "vdb", "align", "cache-ops", "cache-seq", "cache-emb"):
                downstream.setdefault(
                    name,
                    DownstreamStatus(status="down", detail={"error": "aggregate timeout"}),
                )

        aggregate_status = "ok" if all(v.status == "ok" for v in downstream.values()) else "down"
        payload = AggregatedHealthResponse(
            status=aggregate_status,
            service="api",
            downstream=downstream,
        )
        _aggregate_cache["ts"] = time.monotonic()
        _aggregate_cache["payload"] = payload
        return payload
