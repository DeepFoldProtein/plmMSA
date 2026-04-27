"""Shared Prometheus instrumentation for every plmMSA service.

One set of metric definitions — `REQUESTS`, `LATENCY`, `INFLIGHT` — with a
`service` label that discriminates between api / embedding / vdb / align.
Each service registers the same `/metrics` router and attaches a
`MetricsMiddleware(service="...")`.

Worker is non-HTTP; it uses the dedicated `worker_*` metrics plus a small
standalone scrape target — see ``start_worker_metrics_server``.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_log = logging.getLogger(__name__)

# One registry per process is fine — each container is its own process, and
# the `service` label on every metric tells Prometheus who reported what.
REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "plmmsa_http_requests_total",
    "HTTP requests by service, method, route, status.",
    labelnames=("service", "method", "route", "status"),
    registry=REGISTRY,
)
LATENCY = Histogram(
    "plmmsa_http_request_duration_seconds",
    "HTTP request latency by service, method, route.",
    labelnames=("service", "method", "route"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "plmmsa_http_in_flight_requests",
    "In-flight HTTP requests by service, method, route.",
    labelnames=("service", "method", "route"),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Worker-specific metrics. They share the same REGISTRY so a scrape of the
# worker's standalone metrics port gets both the http-style metrics (empty
# for worker) and these, which is fine.
# ---------------------------------------------------------------------------

WORKER_JOBS_PROCESSED = Counter(
    "plmmsa_worker_jobs_processed_total",
    "Jobs drained by the worker loop, by terminal status.",
    labelnames=("status",),  # succeeded | failed | cancelled
    registry=REGISTRY,
)
WORKER_PIPELINE_DURATION = Histogram(
    "plmmsa_worker_pipeline_duration_seconds",
    "Per-job pipeline duration (orchestrator.run wall time).",
    # Wider buckets than HTTP — real MSA jobs run 10-300 s.
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
    registry=REGISTRY,
)
WORKER_QUEUE_DEPTH = Gauge(
    "plmmsa_worker_queue_depth",
    "Most recently observed Redis queue depth (LLEN plmmsa:queue).",
    registry=REGISTRY,
)

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus scrape endpoint. Not rate-limited, not auth-gated —
    scrape from a trusted network only (or front with Cloudflare Access)."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """Record request count / latency / in-flight by (service, method, route, status).

    Routes are taken from ``scope["route"].path`` once FastAPI has resolved
    a route, so ``/v2/msa/{job_id}`` lands in one bucket rather than one
    per job id. Unresolved routes (404s) fall back to the literal path so
    bad-URL spam is still visible.
    """

    def __init__(self, app: ASGIApp, *, service: str) -> None:
        self.app = app
        self.service = service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method") or "GET"
        started = time.perf_counter()
        status_holder = {"status": 0, "route": scope.get("path", "unknown")}

        async def tagged_send(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                status_holder["status"] = int(msg.get("status", 0))
                route = scope.get("route")
                if route is not None and hasattr(route, "path"):
                    status_holder["route"] = route.path
            await send(msg)

        route_label = status_holder["route"]
        INFLIGHT.labels(service=self.service, method=method, route=route_label).inc()
        try:
            await self.app(scope, receive, tagged_send)
        finally:
            duration = time.perf_counter() - started
            route_label = status_holder["route"]
            REQUESTS.labels(
                service=self.service,
                method=method,
                route=route_label,
                status=str(status_holder["status"]),
            ).inc()
            LATENCY.labels(service=self.service, method=method, route=route_label).observe(duration)
            INFLIGHT.labels(service=self.service, method=method, route=route_label).dec()


# ---------------------------------------------------------------------------
# Worker-side entry point
# ---------------------------------------------------------------------------


def start_worker_metrics_server(port: int, *, registry: CollectorRegistry = REGISTRY) -> None:
    """Start a background HTTP server on ``port`` exposing ``/metrics``.

    Worker has no ASGI surface of its own, so it piggybacks on
    ``prometheus_client.start_http_server`` to expose the same registry.
    Binds to 0.0.0.0 inside the container; compose exposes it only on the
    internal ``plmmsa_net`` bridge — not on the host.
    """
    start_http_server(port, registry=registry)
    _log.info("worker: prometheus /metrics server listening on :%d", port)


__all__ = [
    "INFLIGHT",
    "LATENCY",
    "REGISTRY",
    "REQUESTS",
    "WORKER_JOBS_PROCESSED",
    "WORKER_PIPELINE_DURATION",
    "WORKER_QUEUE_DEPTH",
    "MetricsMiddleware",
    "router",
    "start_worker_metrics_server",
]
