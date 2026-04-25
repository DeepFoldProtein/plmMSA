"""Result cache for completed MSA jobs — repurposes the `cache-emb` Redis.

Key layout:
    plmmsa:result:<sha256>   →  JSON-encoded {format, payload, stats}

The key is ``sha256`` of a canonicalized JSON blob built from the
fields of the submit request that *actually* affect the MSA output.
Fields that are request-only (opt-out knobs, audit hints) or that vary
without changing the output are stripped before hashing so independent
submissions of the same scientific query land on the same cache slot.

Canonicalization policy (documented here because it is the cache's
correctness invariant):

- **Sequences** are uppercased and stripped of whitespace. The residue
  alphabet is single-letter; case and whitespace never change the
  biological meaning, and clients occasionally paste with lowercase /
  line breaks.
- **Chain order is preserved.** For paired MSAs, chain A vs chain B
  is not symmetric — the per-chain pairing joins and A3M row order
  depend on the input order. Swapping chains produces a different
  MSA, so they hash to different keys.
- **`query_ids`** affects only the A3M header labels, but those labels
  land in the payload we're caching — so they're part of the key.
- **`force_recompute`** and any other opt-out flags are **dropped**
  before hashing; they influence whether the cache is consulted, not
  what the result looks like.
- **`None` / unset fields** are dropped so two requests that differ
  only in whether they explicitly pass ``None`` vs omit a key still
  collide (they'd produce the same output).
- **Server-resolved fields** (``models``, ``score_model``,
  ``collections``) are included because they land in the submit
  payload after the validator stamps them on. Clients that omit them
  and let the server resolve will collide with clients that pin the
  same values explicitly — which is correct: the orchestrator runs
  the same pipeline either way.

Anything the cache is wrong about will surface as a cache hit
returning the wrong payload, so extending the canonicalization later
is a backwards-compat concern: the cache keyspace is versioned via
``CACHE_VERSION`` below so a bump invalidates all prior entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis

from plmmsa.jobs.models import JobResult

logger = logging.getLogger(__name__)

# Bump when the canonicalization changes in a way that would produce
# different keys for the same request — invalidates every stored entry.
CACHE_VERSION = "v1"
KEY_PREFIX = f"plmmsa:result:{CACHE_VERSION}:"

# Fields copied from the submit payload into the canonical form. Order here
# does not matter (we sort keys in the final serialization); what matters is
# the *set* — adding a field later should also bump CACHE_VERSION.
_CACHED_FIELDS: tuple[str, ...] = (
    "sequences",
    "query_ids",
    "models",
    "paired",
    "output_format",
    "k",
    "aligner",
    "mode",
    "score_model",
    "options",
    "filter_by_score",
    "collections",
)


def _canonicalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a submit payload to the subset that determines the MSA output.

    Strips non-output-affecting fields, normalizes sequences, and drops
    unset / empty values so two semantically equivalent submits collide.
    """
    out: dict[str, Any] = {}
    for key in _CACHED_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            continue
        if key == "sequences" and isinstance(value, list):
            # Uppercase + whitespace strip; preserves chain order.
            value = ["".join(str(s).split()).upper() for s in value]
        if isinstance(value, dict) and not value:
            # Don't let `{}` vs missing produce different keys.
            continue
        out[key] = value
    return out


def cache_key(payload: dict[str, Any]) -> str:
    """Return the deterministic Redis key for `payload`."""
    canonical = _canonicalize(payload)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{digest}"


class ResultCache:
    """Thin async wrapper over a Redis instance holding completed MSAs.

    Missing Redis → all get()s return None, set()s are no-ops; the api
    and worker degrade to non-caching behavior. An explicitly-disabled
    cache (``None`` in place of a client) uses this same degraded path.
    """

    def __init__(self, redis: Redis | None, *, ttl_s: int) -> None:
        self._redis = redis
        self._ttl_s = ttl_s

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    async def get(self, payload: dict[str, Any]) -> JobResult | None:
        if self._redis is None:
            return None
        key = cache_key(payload)
        try:
            raw = await self._redis.get(key)
        except Exception:
            logger.warning("result_cache: GET failed for key=%s", key, exc_info=True)
            return None
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return JobResult.model_validate_json(raw)
        except Exception:
            logger.warning("result_cache: decode failed for key=%s", key, exc_info=True)
            return None

    async def set(self, payload: dict[str, Any], result: JobResult) -> None:
        if self._redis is None:
            return
        key = cache_key(payload)
        try:
            await self._redis.set(key, result.model_dump_json(), ex=self._ttl_s)
        except Exception:
            logger.warning("result_cache: SET failed for key=%s", key, exc_info=True)


__all__ = ["CACHE_VERSION", "KEY_PREFIX", "ResultCache", "cache_key"]
