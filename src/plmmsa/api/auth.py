from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from plmmsa.admin.tokens import TokenStore
from plmmsa.errors import ErrorCode, PlmMSAError

_bearer = HTTPBearer(auto_error=False)


async def get_token_store(request: Request) -> TokenStore:
    """FastAPI dependency returning the app-local `TokenStore`.

    The store is constructed in `create_app` and attached to
    `app.state.token_store`. Tests replace it by assigning a new store on the
    same attribute before making requests (see `tests/conftest.py`).
    """
    return request.app.state.token_store


async def require_admin_token(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    store: Annotated[TokenStore, Depends(get_token_store)],
) -> None:
    """Gate for privileged routes.

    Accepts either:

    - The bootstrap `ADMIN_TOKEN` env var (so the stack has a usable
      credential before any tokens have been minted). Typical use: call
      `POST /admin/tokens` once to mint per-client tokens, then stop using
      the bootstrap token for day-to-day requests.
    - A live token in the `TokenStore` (Redis-backed, SHA-256 hashed; minted
      via `/admin/tokens`).
    """
    if creds is None or not creds.credentials:
        raise PlmMSAError(
            "Authorization bearer token required.",
            code=ErrorCode.AUTH_MISSING,
            http_status=401,
        )
    candidate = creds.credentials

    bootstrap = os.environ.get("ADMIN_TOKEN", "")
    if bootstrap and candidate == bootstrap:
        return

    record = await store.verify(candidate)
    if record is not None:
        return

    raise PlmMSAError(
        "Invalid API token.",
        code=ErrorCode.AUTH_INVALID,
        http_status=401,
    )
