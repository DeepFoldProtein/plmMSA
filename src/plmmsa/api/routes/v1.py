from __future__ import annotations

from fastapi import APIRouter

from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError

router = APIRouter(tags=["v1-legacy"])


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def v1_gone(path: str) -> None:
    settings = get_settings()
    raise PlmMSAError(
        f"The /v1/ API was sunset on {settings.api.v1_sunset_date}. Use /v2/ instead.",
        code=ErrorCode.GONE,
        http_status=410,
        detail={"path": path, "successor": "/v2/"},
    )
