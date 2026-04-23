"""Prometheus /metrics — request counter, latency histogram, in-flight gauge.

Exposed via a bare FastAPI router (`plmmsa.api.metrics.router`) so the same
module can be included by sidecars (embedding, vdb, align) that want the
same per-service counters without pulling in the whole middleware stack.

The `MetricsMiddleware` records on the api app; sidecars can add their own
instance if they want per-service labels.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# A dedicated registry avoids multiprocessing issues (uvicorn with >1 worker
# would otherwise fork a shared default registry — we rely on Prometheus
# scraping one replica per uvicorn worker and aggregating upstream).
REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "plmmsa_http_requests_total",
    "HTTP requests by method, route, status.",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)
LATENCY = Histogram(
    "plmmsa_http_request_duration_seconds",
    "HTTP request latency by method + route.",
    labelnames=("method", "route"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=REGISTRY,
)
INFLIGHT = Gauge(
    "plmmsa_http_in_flight_requests",
    "In-flight HTTP requests by method + route.",
    labelnames=("method", "route"),
    registry=REGISTRY,
)

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware:
    """Record request count / latency / in-flight by (method, route, status).

    Routes are taken from `scope["route"].path` when FastAPI resolved a route
    so `/v2/msa/{job_id}` lands in one bucket rather than one per job id. If
    the route didn't resolve (404), we fall back to the literal path to still
    get useful signal.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

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
        INFLIGHT.labels(method=method, route=route_label).inc()
        try:
            await self.app(scope, receive, tagged_send)
        finally:
            duration = time.perf_counter() - started
            route_label = status_holder["route"]
            REQUESTS.labels(
                method=method,
                route=route_label,
                status=str(status_holder["status"]),
            ).inc()
            LATENCY.labels(method=method, route=route_label).observe(duration)
            INFLIGHT.labels(method=method, route=route_label).dec()


__all__ = ["INFLIGHT", "LATENCY", "REGISTRY", "REQUESTS", "MetricsMiddleware", "router"]
