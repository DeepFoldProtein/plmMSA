from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from plmmsa import __version__
from plmmsa.admin.routes import router as admin_router
from plmmsa.admin.tokens import TokenStore
from plmmsa.api.health import router as health_router
from plmmsa.api.metrics import MetricsMiddleware
from plmmsa.api.metrics import router as metrics_router
from plmmsa.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    configure_logging,
)
from plmmsa.api.routes.v1 import router as v1_router
from plmmsa.api.routes.v2 import router as v2_router
from plmmsa.config import get_settings
from plmmsa.errors import PlmMSAError


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Drain cleanly on SIGTERM: close Redis connections so in-flight
    responses don't race a torn-down pool. Uvicorn already handles the
    "stop accepting + drain in-flight" loop; we just add the cleanup.
    """
    yield
    log = logging.getLogger("plmmsa.lifespan")
    client = getattr(app.state, "ratelimit_redis", None)
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            log.exception("lifespan: ratelimit redis close failed")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.logging.level, json_format=settings.logging.json_format)
    app = FastAPI(
        title="plmMSA",
        version=__version__,
        description="PLM-based MSA server.",
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
        lifespan=_lifespan,
    )

    # Shared `TokenStore` + rate-limit Redis live on `app.state` so tests can
    # swap them out by assigning to `app.state.token_store` /
    # `app.state.ratelimit_redis` without touching globals.
    cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
    redis_client = Redis.from_url(cache_url, decode_responses=False)
    app.state.token_store = TokenStore(redis_client)
    app.state.ratelimit_redis = redis_client

    # Middleware runs outside-in on the request, inside-out on the response.
    # Register them in the order we want them to see the request, starting
    # from the innermost. Stack built here (outermost first on the wire):
    #   RequestContext → RateLimit → BodySizeLimit → CORS → app
    # - RequestContext stamps X-Request-ID + client_ip state, so everything
    #   downstream (incl. rate-limit denial logs and audit events) can see
    #   a stable id.
    # - RateLimit uses that state, gates the request, and stashes the
    #   resolved token_id on scope.state for audit logs.
    # - BodySizeLimit rejects oversized bodies before JSON parsing.
    # - CORS is innermost so preflights from rejected (oversized) requests
    #   don't need CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.limits.max_body_bytes)
    app.add_middleware(
        RateLimitMiddleware,
        per_ip_rpm=settings.ratelimit.per_ip_rpm,
        per_token_rpm_default=settings.ratelimit.per_token_rpm,
        bootstrap_token=os.environ.get("ADMIN_TOKEN") or None,
        exempt_paths=("/healthz", "/readyz", "/metrics", "/v2/version"),
    )
    app.add_middleware(RequestContextMiddleware, header=settings.logging.request_id_header)

    @app.exception_handler(PlmMSAError)
    async def plmmsa_error_handler(_: Request, exc: PlmMSAError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
            headers=exc.headers,
        )

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(v1_router, prefix="/v1")
    app.include_router(v2_router, prefix="/v2")
    # Admin routes — intentionally co-located under the same FastAPI app so
    # the internal operator can reach them via localhost, but NEVER route
    # `/admin/*` through the public Cloudflare tunnel hostname (see
    # docs/cloudflare-tunnel.md).
    app.include_router(admin_router)
    return app


app = create_app()
