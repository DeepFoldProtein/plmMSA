from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from plmmsa import __version__
from plmmsa.admin.routes import router as admin_router
from plmmsa.admin.tokens import TokenStore
from plmmsa.api.health import router as health_router
from plmmsa.api.routes.v1 import router as v1_router
from plmmsa.api.routes.v2 import router as v2_router
from plmmsa.config import get_settings
from plmmsa.errors import PlmMSAError


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="plmMSA",
        version=__version__,
        description="PLM-based MSA server.",
        openapi_url="/openapi.json" if settings.api.openapi_public else None,
        docs_url="/docs" if settings.api.openapi_public else None,
        redoc_url="/redoc" if settings.api.openapi_public else None,
    )

    # Shared `TokenStore` lives on `app.state` so tests can swap it out by
    # assigning to `app.state.token_store` without touching globals.
    cache_url = os.environ.get("CACHE_URL", "redis://cache:6379")
    app.state.token_store = TokenStore(Redis.from_url(cache_url, decode_responses=False))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(PlmMSAError)
    async def plmmsa_error_handler(_: Request, exc: PlmMSAError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.as_response().model_dump(mode="json"),
        )

    app.include_router(health_router)
    app.include_router(v1_router, prefix="/v1")
    app.include_router(v2_router, prefix="/v2")
    # Admin routes — intentionally co-located under the same FastAPI app so
    # the internal operator can reach them via localhost, but NEVER route
    # `/admin/*` through the public Cloudflare tunnel hostname (see
    # docs/cloudflare-tunnel.md).
    app.include_router(admin_router)
    return app


app = create_app()
