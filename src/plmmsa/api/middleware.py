"""HTTP middleware: body-size cap, request-id propagation, structured
access + audit logs, and rate limiting (per-IP and per-token).

The middleware stack is attached in `plmmsa.api.create_app` (order matters —
see the comments next to `app.add_middleware` there).

Why ASGI-style middleware for the body-size check:
- `Content-Length`-based short-circuits miss chunked / unknown-length uploads.
- FastAPI has no first-class "max body" knob, and Starlette's BaseHTTPMiddleware
  buffers the full body into memory before handing it to the next layer,
  which defeats the purpose.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from plmmsa.admin.tokens import TokenStore
from plmmsa.errors import ErrorCode, ErrorResponse

_ACCESS_LOG = logging.getLogger("plmmsa.access")
_AUDIT_LOG = logging.getLogger("plmmsa.audit")


def _client_ip(scope: Scope) -> str:
    # Trust `X-Forwarded-For` from Cloudflare's tunnel — the real public IP is
    # in the left-most token. If it's missing (direct localhost / tests), fall
    # back to the peer.
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            txt = value.decode("latin-1")
            return txt.split(",", 1)[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


def _header(scope: Scope, name: bytes) -> str | None:
    for n, v in scope.get("headers", []):
        if n == name:
            return v.decode("latin-1")
    return None


async def _send_error(
    send: Send,
    *,
    status: int,
    code: ErrorCode,
    message: str,
    detail: dict[str, Any] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = ErrorResponse(code=code, message=message, detail=detail).model_dump_json().encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body exceeds `max_bytes`.

    Runs before route parsing so oversized payloads never hit JSON decoding.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await _send_error(
                        send,
                        status=413,
                        code=ErrorCode.INVALID_FASTA,
                        message=f"Request body exceeds {self.max_bytes} bytes.",
                        detail={"max_bytes": self.max_bytes, "declared": int(declared)},
                    )
                    return
            except ValueError:
                pass

        received = 0
        max_bytes = self.max_bytes
        over = False

        async def guarded_receive() -> Message:
            nonlocal received, over
            msg = await receive()
            if msg["type"] == "http.request":
                body = msg.get("body", b"") or b""
                received += len(body)
                if received > max_bytes:
                    over = True
            return msg

        sent_started = False

        async def guarded_send(msg: Message) -> None:
            nonlocal sent_started
            if msg["type"] == "http.response.start":
                sent_started = True
            if over and not sent_started:
                await _send_error(
                    send,
                    status=413,
                    code=ErrorCode.INVALID_FASTA,
                    message=f"Request body exceeds {max_bytes} bytes.",
                    detail={"max_bytes": max_bytes},
                )
                return
            await send(msg)

        await self.app(scope, guarded_receive, guarded_send)


class RequestContextMiddleware:
    """Stamp every request with an X-Request-ID and emit a structured access
    log line including the resolved client IP, status, duration, and
    (if present) the admin-token id used to authorize the call.

    Downstream sidecars (embedding, vdb, align) re-emit the header on their
    outgoing responses; the orchestrator forwards it on internal httpx calls.
    """

    def __init__(self, app: ASGIApp, *, header: str = "X-Request-ID") -> None:
        self.app = app
        self.header_lc = header.lower()
        self.header_bytes = header.encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = _header(scope, self.header_lc.encode("latin-1"))
        request_id = supplied if supplied else uuid.uuid4().hex
        scope.setdefault("state", {})
        state = scope["state"]  # pyright: ignore[reportGeneralTypeIssues]
        state["request_id"] = request_id
        state["client_ip"] = _client_ip(scope)

        started = time.perf_counter()
        status_holder: dict[str, int] = {"status": 0}

        async def tagged_send(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                status_holder["status"] = int(msg.get("status", 0))
                headers = list(msg.get("headers", []))
                headers.append((self.header_bytes, request_id.encode("latin-1")))
                msg = {**msg, "headers": headers}
            await send(msg)

        try:
            await self.app(scope, receive, tagged_send)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            _ACCESS_LOG.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_holder["status"],
                    "duration_ms": round(duration_ms, 2),
                    "client_ip": state.get("client_ip"),
                    "token_id": state.get("token_id"),
                },
            )


async def _consume_rate_limit(
    redis: Any,
    *,
    key: str,
    limit: int,
    window_s: int = 60,
) -> tuple[bool, int]:
    """Fixed-window counter. Returns `(allowed, retry_after_seconds)`.

    Chosen over sliding-log for simplicity: a single INCR + EXPIRE per
    request, no ZSET scan. The ±window_s edge artifact is tolerable for
    abuse protection where the actual knob is "don't flood".
    """
    bucket = int(time.time()) // window_s
    redis_key = f"rl:{key}:{bucket}"
    pipe = redis.pipeline()
    pipe.incr(redis_key)
    pipe.expire(redis_key, window_s * 2)
    result = await pipe.execute()
    count = int(result[0])
    if count > limit:
        retry = window_s - (int(time.time()) % window_s)
        return False, max(retry, 1)
    return True, 0


class RateLimitMiddleware:
    """Two-layer rate limit: per-token (if Bearer is present + resolvable) and
    per-IP (always). The per-token layer uses the TokenRecord's
    `rate_limit_rpm` override, falling back to `settings.ratelimit.per_token_rpm`.

    Both layers share a single cache-ops Redis. The redis client + token
    store are read from `app.state` at request time rather than captured at
    construction, so tests can swap them by assigning to
    `app.state.ratelimit_redis` / `app.state.token_store` without re-building
    the middleware stack.

    Paths in `exempt_paths` bypass the limiter entirely (health checks,
    Prometheus scrape). The bootstrap ADMIN_TOKEN is NOT exempt — abuse of a
    leaked bootstrap still needs a ceiling.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        per_ip_rpm: int,
        per_token_rpm_default: int,
        bootstrap_token: str | None,
        exempt_paths: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.per_ip_rpm = per_ip_rpm
        self.per_token_rpm_default = per_token_rpm_default
        self.bootstrap_token = bootstrap_token or None
        self.exempt_paths = exempt_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(p) for p in self.exempt_paths):
            await self.app(scope, receive, send)
            return

        app_state = scope["app"].state  # pyright: ignore[reportGeneralTypeIssues]
        redis = app_state.ratelimit_redis
        token_store: TokenStore = app_state.token_store

        state = scope.setdefault("state", {})
        ip = state.get("client_ip") or _client_ip(scope)

        auth = _header(scope, b"authorization") or ""
        token_id: str | None = None
        token_limit = self.per_token_rpm_default
        if auth.lower().startswith("bearer "):
            raw = auth.split(None, 1)[1].strip()
            if raw:
                if self.bootstrap_token and raw == self.bootstrap_token:
                    token_id = "bootstrap"
                else:
                    record = await token_store.verify(raw)
                    if record is not None:
                        token_id = record.id
                        if record.rate_limit_rpm:
                            token_limit = record.rate_limit_rpm
        if token_id is not None:
            allowed, retry = await _consume_rate_limit(
                redis,
                key=f"token:{token_id}",
                limit=token_limit,
            )
            if not allowed:
                await _send_error(
                    send,
                    status=429,
                    code=ErrorCode.RATE_LIMITED,
                    message="Per-token rate limit exceeded.",
                    detail={"limit_rpm": token_limit, "scope": "token"},
                    extra_headers=[(b"retry-after", str(retry).encode())],
                )
                return
            state["token_id"] = token_id

        allowed, retry = await _consume_rate_limit(
            redis,
            key=f"ip:{ip}",
            limit=self.per_ip_rpm,
        )
        if not allowed:
            await _send_error(
                send,
                status=429,
                code=ErrorCode.RATE_LIMITED,
                message="Per-IP rate limit exceeded.",
                detail={"limit_rpm": self.per_ip_rpm, "scope": "ip"},
                extra_headers=[(b"retry-after", str(retry).encode())],
            )
            return

        await self.app(scope, receive, send)


class JSONLogFormatter(logging.Formatter):
    """Emit one JSON object per log line, carrying whatever structured fields
    the caller attached via `extra=`. Prefer this over `%(request_id)s`-in-
    format because it gracefully handles missing fields.
    """

    _STD_FIELDS: ClassVar[set[str]] = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._STD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(*, level: str = "INFO", json_format: bool = True) -> None:
    """Root logging config shared by api + worker + embedding + vdb + align.

    Idempotent — we clear existing handlers so repeated calls (e.g. in tests
    under `uvicorn.run`) don't double-emit.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(level.upper())


def audit_event(event: str, **fields: Any) -> None:
    """Append-only privileged-action log. Callers supply fields like
    `route="/v2/msa"`, `token_id=...`, `job_id=...`.

    Routed through the `plmmsa.audit` logger so operators can fan it out to a
    dedicated file / sink via standard logging config.
    """
    _AUDIT_LOG.info(event, extra=fields)


__all__ = [
    "BodySizeLimitMiddleware",
    "JSONLogFormatter",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
    "audit_event",
    "configure_logging",
]


# Starlette's BaseHTTPMiddleware is deliberately avoided; type alias kept to
# keep the module self-describing for anyone reading imports.
MiddlewareCallable = Callable[[Scope, Receive, Send], Awaitable[None]]
