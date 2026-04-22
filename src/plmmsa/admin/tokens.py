from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from typing import Any

from pydantic import BaseModel
from redis.asyncio import Redis

_PREFIX = "admintoken"


class TokenRecord(BaseModel):
    """Metadata about a minted admin token. The token plaintext is not stored."""

    id: str
    label: str
    created_at: float
    expires_at: float | None = None
    revoked: bool = False
    rate_limit_rpm: int | None = None


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenStore:
    """Redis-backed token directory.

    Layout:
      {prefix}:rec:{id}     → TokenRecord JSON
      {prefix}:hash:{sha256} → id (constant-time lookup from token plaintext)
      {prefix}:all           → SET of all token ids

    Tokens are never stored in plaintext; verification hashes the incoming
    token and looks it up by its digest.
    """

    def __init__(self, redis: Redis, *, prefix: str = _PREFIX) -> None:
        self._redis = redis
        self._prefix = prefix

    def _rec_key(self, tid: str) -> str:
        return f"{self._prefix}:rec:{tid}"

    def _hash_key(self, digest: str) -> str:
        return f"{self._prefix}:hash:{digest}"

    def _index_key(self) -> str:
        return f"{self._prefix}:all"

    async def mint(
        self,
        *,
        label: str,
        expires_at: float | None = None,
        rate_limit_rpm: int | None = None,
    ) -> tuple[str, TokenRecord]:
        """Generate a new token. Returns `(plaintext, record)` — plaintext is
        shown once and never retrievable again."""
        token = secrets.token_urlsafe(32)
        tid = str(uuid.uuid4())
        record = TokenRecord(
            id=tid,
            label=label,
            created_at=time.time(),
            expires_at=expires_at,
            rate_limit_rpm=rate_limit_rpm,
        )
        await self._redis.set(self._rec_key(tid), record.model_dump_json())  # pyright: ignore[reportGeneralTypeIssues]
        await self._redis.set(self._hash_key(_sha(token)), tid)  # pyright: ignore[reportGeneralTypeIssues]
        await self._redis.sadd(self._index_key(), tid)  # pyright: ignore[reportGeneralTypeIssues]
        return token, record

    async def verify(self, token: str) -> TokenRecord | None:
        """Return the record if `token` is valid, non-revoked, non-expired."""
        tid = await self._redis.get(self._hash_key(_sha(token)))
        if tid is None:
            return None
        if isinstance(tid, bytes):
            tid = tid.decode("utf-8")
        raw = await self._redis.get(self._rec_key(tid))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        record = TokenRecord.model_validate_json(raw)
        if record.revoked:
            return None
        if record.expires_at is not None and time.time() > record.expires_at:
            return None
        return record

    async def get(self, tid: str) -> TokenRecord | None:
        raw = await self._redis.get(self._rec_key(tid))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return TokenRecord.model_validate_json(raw)

    async def list(self) -> list[TokenRecord]:
        ids_raw: Any = await self._redis.smembers(self._index_key())  # pyright: ignore[reportGeneralTypeIssues]
        out: list[TokenRecord] = []
        for id_raw in ids_raw:
            tid = id_raw.decode("utf-8") if isinstance(id_raw, bytes) else id_raw
            record = await self.get(tid)
            if record is not None:
                out.append(record)
        out.sort(key=lambda r: r.created_at)
        return out

    async def revoke(self, tid: str) -> TokenRecord | None:
        """Mark the token revoked. The record is preserved for audit; future
        `verify` calls return None."""
        record = await self.get(tid)
        if record is None:
            return None
        record.revoked = True
        await self._redis.set(self._rec_key(tid), record.model_dump_json())  # pyright: ignore[reportGeneralTypeIssues]
        return record
