from __future__ import annotations

import time as _time
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, Response
from pydantic import BaseModel, Field

from plmmsa.admin.tokens import TokenRecord, TokenStore
from plmmsa.api.auth import get_token_store, require_admin_token
from plmmsa.api.middleware import audit_event
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)


class MintTokenRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    expires_at: float | None = Field(
        None, description="Unix timestamp after which the token is invalid."
    )
    rate_limit_rpm: int | None = Field(None, ge=1)


class MintTokenResponse(BaseModel):
    token: str = Field(..., description="Plaintext token. Shown ONCE — store it now.")
    record: TokenRecord


class TokenListResponse(BaseModel):
    tokens: list[TokenRecord]


@router.post("/tokens", response_model=MintTokenResponse)
async def mint_token(
    req: MintTokenRequest,
    request: Request,
    store: Annotated[TokenStore, Depends(get_token_store)],
) -> MintTokenResponse:
    # If the caller doesn't specify `expires_at`, apply
    # `settings.api.default_token_ttl_s` so non-expiring tokens aren't the
    # default shape. A request sending `expires_at=0` opts in to "no expiry"
    # explicitly and is honored.
    settings = get_settings()
    expires_at = req.expires_at
    if expires_at is None and settings.api.default_token_ttl_s > 0:
        expires_at = _time.time() + settings.api.default_token_ttl_s
    elif expires_at == 0:
        expires_at = None
    token, record = await store.mint(
        label=req.label,
        expires_at=expires_at,
        rate_limit_rpm=req.rate_limit_rpm,
    )
    audit_event(
        "admin.token.mint",
        actor_token_id=getattr(request.state, "token_id", None),
        request_id=getattr(request.state, "request_id", None),
        client_ip=getattr(request.state, "client_ip", None),
        minted_token_id=record.id,
        label=record.label,
        expires_at=record.expires_at,
        rate_limit_rpm=record.rate_limit_rpm,
    )
    return MintTokenResponse(token=token, record=record)


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(
    store: Annotated[TokenStore, Depends(get_token_store)],
) -> TokenListResponse:
    return TokenListResponse(tokens=await store.list())


@router.delete("/tokens/{token_id}", status_code=204)
async def revoke_token(
    request: Request,
    store: Annotated[TokenStore, Depends(get_token_store)],
    token_id: Annotated[str, Path(min_length=1)],
) -> Response:
    record = await store.revoke(token_id)
    if record is None:
        raise PlmMSAError(
            f"Token {token_id} not found.",
            code=ErrorCode.JOB_NOT_FOUND,
            http_status=404,
            detail={"token_id": token_id},
        )
    audit_event(
        "admin.token.revoke",
        actor_token_id=getattr(request.state, "token_id", None),
        request_id=getattr(request.state, "request_id", None),
        client_ip=getattr(request.state, "client_ip", None),
        revoked_token_id=token_id,
    )
    return Response(status_code=204)
