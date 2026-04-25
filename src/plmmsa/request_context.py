"""Shared request-context plumbing — lets every service (api, embedding,
vdb, align, worker) stamp a stable X-Request-ID on its access logs and
forward it on outgoing HTTP calls.

The public surface is small:

- ``REQUEST_ID`` ContextVar + ``current_request_id()`` / ``bind_request_id()``
  helpers. Call ``bind_request_id(rid)`` on a worker job entry and every
  downstream ``httpx`` client built via ``Orchestrator._new_client`` picks
  it up via the context.
- ``RequestContextMiddleware`` — ASGI middleware for sidecar FastAPI apps
  (embedding, vdb, align). Reads the incoming ``X-Request-ID``, falls back
  to a fresh uuid if absent, puts it on scope.state + ContextVar, echoes
  it on the response headers, and emits a structured access-log line.

The api's richer middleware in ``plmmsa.api.middleware`` layers on top of
this (bodysize, rate limits, audit logs) — it does not need to import
from here because its own stamping already sets the same ContextVar via
the ``bind_request_id`` call path. Sidecars import from here directly.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID: ContextVar[str | None] = ContextVar("plmmsa_request_id", default=None)

_HEADER = "X-Request-ID"
_HEADER_BYTES = _HEADER.encode("latin-1")
_HEADER_LC_BYTES = _HEADER.lower().encode("latin-1")


def current_request_id() -> str | None:
    """Return the X-Request-ID bound to the current async context, or None."""
    return REQUEST_ID.get()


def bind_request_id(rid: str | None) -> None:
    """Set the X-Request-ID for the current async context.

    Called from worker (on job claim) and api middleware (on request entry).
    Downstream ``httpx`` clients pick it up via
    ``httpx_headers_with_request_id`` or by constructing the client with
    ``headers=httpx_headers_with_request_id()``.
    """
    REQUEST_ID.set(rid)


def httpx_headers_with_request_id(
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return an httpx headers dict that carries the current request id.

    Never overwrites an explicit caller-supplied ``X-Request-ID``.
    """
    out = dict(existing) if existing else {}
    rid = current_request_id()
    if rid and _HEADER not in out:
        out[_HEADER] = rid
    return out


def _read_header(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == _HEADER_LC_BYTES:
            return value.decode("latin-1")
    return None


class RequestContextMiddleware:
    """Sidecar-service version of the api's request-context middleware.

    Lighter than the api variant: no token/rate-limit/audit plumbing, just
    (1) adopt or generate X-Request-ID, (2) bind it to the ContextVar so
    logs and downstream httpx calls can correlate, (3) echo it on the
    response, and (4) emit one structured access-log line per request.

    ``service`` is the label that shows up on the access-log line
    (``"embedding"`` / ``"vdb"`` / ``"align"``), and on the logger name
    (``"plmmsa.access.{service}"``) so operators can filter by service.
    """

    def __init__(self, app: ASGIApp, *, service: str) -> None:
        self.app = app
        self.service = service
        self._log = logging.getLogger(f"plmmsa.access.{service}")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = _read_header(scope)
        request_id = supplied if supplied else uuid.uuid4().hex
        scope.setdefault("state", {})
        state: dict[str, Any] = scope["state"]  # pyright: ignore[reportGeneralTypeIssues]
        state["request_id"] = request_id
        bind_request_id(request_id)

        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 0}

        async def tagged_send(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                status_holder["status"] = int(msg.get("status", 0))
                headers = list(msg.get("headers", []))
                headers.append((_HEADER_BYTES, request_id.encode("latin-1")))
                msg = {**msg, "headers": headers}
            await send(msg)

        try:
            await self.app(scope, receive, tagged_send)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self._log.info(
                "request",
                extra={
                    "service": self.service,
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_holder["status"],
                    "duration_ms": round(duration_ms, 2),
                },
            )


__all__ = [
    "REQUEST_ID",
    "RequestContextMiddleware",
    "bind_request_id",
    "current_request_id",
    "httpx_headers_with_request_id",
]
