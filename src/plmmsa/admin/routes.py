from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response
from pydantic import BaseModel, Field

from plmmsa.admin.tokens import TokenRecord, TokenStore
from plmmsa.api.auth import get_token_store, require_admin_token
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
    store: Annotated[TokenStore, Depends(get_token_store)],
) -> MintTokenResponse:
    token, record = await store.mint(
        label=req.label,
        expires_at=req.expires_at,
        rate_limit_rpm=req.rate_limit_rpm,
    )
    return MintTokenResponse(token=token, record=record)


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(
    store: Annotated[TokenStore, Depends(get_token_store)],
) -> TokenListResponse:
    return TokenListResponse(tokens=await store.list())


@router.delete("/tokens/{token_id}", status_code=204)
async def revoke_token(
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
    return Response(status_code=204)
