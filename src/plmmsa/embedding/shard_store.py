"""Precomputed per-residue embedding shard reader.

Legacy plmMSA stacks generated PLM embeddings once offline and parked the
results on disk; the hosted service then served them instead of re-running
the model. The ProtT5 shards at `/gpfs/database/milvus/datasets/uniref50_t5/datasets`
are the canonical example: ~60 M UniRef50 sequences, one `.pt` file per
sequence, with an SQLite index (`index.db`) mapping filename → shard
directory.

This reader is deliberately PLM-agnostic — nothing here cares that the
embeddings came from ProtT5. If we ever build an equivalent store for Ankh
or ESM the same class works (point it at a different root + index).

Design points worth knowing:
- Synchronous API. sqlite3 + torch.load aren't async; FastAPI handlers
  should call `fetch` via `asyncio.to_thread` to keep the event loop free.
- One sqlite connection per call (opened read-only, immutable). The index
  is cheap to open, avoids cross-thread-owner gotchas, and doesn't hold a
  WAL lock across requests.
- Missing files degrade to "miss", never raise. Stale index rows happen
  when shards are rebuilt; we don't want a 500 for those.
- Dim mismatch DOES raise. A shard claiming dim 1024 but returning 768
  would silently feed wrong-sized vectors into the aligner — worse than
  a loud failure.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import re
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ShardDimMismatchError(RuntimeError):
    """Raised when a loaded shard tensor disagrees with the configured dim.

    Hard error on purpose — the alternative is silently feeding the aligner
    vectors of the wrong rank, which corrupts results without any signal.
    """


# Backward-compat alias — earlier drafts used `ShardDimMismatch` without the
# N818 `Error` suffix. Remove after downstream imports are updated.
ShardDimMismatch = ShardDimMismatchError


_FRAGMENT_RE = re.compile(r"_F(\d+)$")


class ShardStore:
    """Read-only shard-backed embedding cache.

    `fetch(ids)` returns `(found, missing)` where `found` maps each present
    id to an `np.ndarray[L, dim]` (fp32) and `missing` is the list of ids
    that were neither in the SQLite index nor in any fallback directory.

    Path-index backends
    -------------------
    Two index sources, tried in order on each lookup:

    1. **Redis MGET** (preferred when `redis_url` is set). Keys like
       `shard:prott5:<bare_id>` map to the folder name. On a local
       network this resolves 1000s of ids in single-digit ms — vs.
       ~13 s for the same workload against a `/gpfs`-hosted sqlite.
       Populate once from the sqlite index via
       `python -m plmmsa.tools.build_shard_index`.

    2. **SQLite `index.db`** — fallback, also the source of truth the
       Redis index was built from. Kept for hosts that haven't run the
       populator yet and for disaster recovery.

    Fallback directories still apply on top of whichever index answers
    (or both: if Redis says "folder 300" but the file isn't actually
    there, we still try fallback dirs before declaring a miss).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        index_db: str | Path | None = None,
        fallback_dirs: Iterable[str | Path] = (),
        dim: int = 1024,
        id_prefix_strip: str = "UniRef50_",
        redis_url: str | None = None,
        redis_key_prefix: str = "shard:prott5:",
    ) -> None:
        self.root = Path(root)
        self.dim = dim
        self._id_prefix_strip = id_prefix_strip
        self._index_db = Path(index_db) if index_db else self.root / "index.db"
        self._fallback_dirs: tuple[Path, ...] = tuple(
            (self.root / d) if not Path(d).is_absolute() else Path(d) for d in fallback_dirs
        )
        # Per-thread handles (unused today, reserved if we switch to pooling).
        self._tls = threading.local()
        # Redis path-index config. Client is lazily constructed on first
        # async call so sync tests that never touch `aresolve_paths`
        # don't pay import cost for redis.asyncio.
        self._redis_url = redis_url
        self._redis_key_prefix = redis_key_prefix
        self._redis_client: Any | None = None

        if not self.root.is_dir():
            logger.warning(
                "shard_store: root %s does not exist — fetch will return all misses",
                self.root,
            )
        elif not self._index_db.is_file():
            logger.warning(
                "shard_store: index %s missing — all lookups will fall back to "
                "fallback_dirs + direct filename probe",
                self._index_db,
            )
        else:
            logger.info(
                "shard_store: root=%s index=%s fallback_dirs=%s dim=%d redis=%s",
                self.root,
                self._index_db,
                [str(d) for d in self._fallback_dirs],
                self.dim,
                self._redis_url or "<disabled>",
            )

    # --- public ---------------------------------------------------------------

    def fetch(self, ids: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """Batch-resolve UniRef50 ids, sequentially.

        Blocks on sqlite + disk; callers inside an event loop should wrap
        this in `asyncio.to_thread`. Kept for simple sync callers; the
        embedding service uses `resolve_paths` + per-file `load_tensor`
        through an asyncio.gather fanout for real parallelism.
        """
        if not ids:
            return {}, []
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []

        resolved = self.resolve_paths(ids)
        for rid, path in resolved:
            if path is None:
                missing.append(rid)
                continue
            arr = self.load_tensor(path)
            if arr is None:
                missing.append(rid)
                continue
            found[rid] = arr
        return found, missing

    def resolve_paths(self, ids: list[str]) -> list[tuple[str, Path | None]]:
        """Synchronous lookup — sqlite + fallback dirs. Callers inside an
        event loop should prefer `aresolve_paths`, which uses Redis MGET
        (~300x faster than sqlite on /gpfs) when `redis_url` is set.
        """
        if not ids:
            return []
        index_by_filename: dict[str, str] = {}
        if self.root.is_dir() and self._index_db.is_file():
            index_by_filename = self._index_lookup_batch([self._filename_for(i) for i in ids])
        out: list[tuple[str, Path | None]] = []
        for rid in ids:
            filename = self._filename_for(rid)
            out.append((rid, self._resolve_path(filename, index_by_filename)))
        return out

    async def aresolve_paths(self, ids: list[str]) -> list[tuple[str, Path | None]]:
        """Async path lookup — Redis MGET + fallback-dir scan.

        SQLite is **not** consulted at runtime (it's too slow on /gpfs).
        The sqlite `index.db` remains the build-time source of truth,
        pumped into Redis via `python -m plmmsa.tools.build_shard_index`.
        Ids Redis doesn't know still get a `_fallback_dirs` probe before
        being declared a miss.

        Returns `(id, path_or_None)` in input order.
        """
        if not ids:
            return []

        index_by_filename: dict[str, str] = {}
        if self._redis_url:
            index_by_filename = await self._redis_lookup_batch(ids)
        elif self._index_db.is_file():
            # No Redis configured — legacy sync sqlite path, running in
            # a worker thread so it doesn't block the event loop. This
            # path is slow on /gpfs and exists only for bootstrap (pre-
            # populate) and unit tests.
            index_by_filename = await asyncio.to_thread(
                self._index_lookup_batch,
                [self._filename_for(i) for i in ids],
            )

        out: list[tuple[str, Path | None]] = []
        for rid in ids:
            filename = self._filename_for(rid)
            out.append((rid, self._resolve_path(filename, index_by_filename)))
        return out

    async def _redis_lookup_batch(self, ids: list[str]) -> dict[str, str]:
        """MGET `shard:prott5:<bare_id>` → folder. Returns a
        filename-keyed map matching `_index_lookup_batch`'s contract so
        the two paths are interchangeable.

        Failures are swallowed with a WARN — callers fall through to
        sqlite. The embedding service never wants a Redis hiccup to
        turn into a job failure.
        """
        try:
            client = await self._get_async_redis()
        except Exception as exc:
            logger.warning("shard_store: async redis client unavailable: %s", exc)
            return {}

        bare_ids = [self._bare_id(i) for i in ids]
        keys = [f"{self._redis_key_prefix}{b}" for b in bare_ids]
        try:
            values = await client.mget(keys)
        except Exception as exc:
            logger.warning("shard_store: redis MGET failed: %s", exc)
            return {}

        out: dict[str, str] = {}
        for bare, val in zip(bare_ids, values, strict=True):
            if val is None:
                continue
            folder = val.decode() if isinstance(val, bytes) else str(val)
            out[f"{bare}.pt"] = folder
        return out

    async def _get_async_redis(self) -> Any:
        """Lazy async-redis client construction. Kept on the instance so
        one connection pool is reused across requests."""
        if self._redis_client is None:
            if not self._redis_url:
                raise RuntimeError("ShardStore.redis_url is unset; cannot construct async client")
            # Import lazily — keeps `redis.asyncio` out of cold paths.
            import redis.asyncio as aioredis

            self._redis_client = aioredis.Redis.from_url(
                self._redis_url,
                decode_responses=False,
            )
        return self._redis_client

    def _bare_id(self, uniref_id: str) -> str:
        """Strip the `UniRef50_` prefix (if present) to match how Redis
        keys + shard filenames are stored."""
        if self._id_prefix_strip and uniref_id.startswith(self._id_prefix_strip):
            return uniref_id[len(self._id_prefix_strip) :]
        return uniref_id

    def load_tensor(self, path: Path) -> np.ndarray | None:
        """Public alias for the internal `_load_tensor`. One file read +
        decode; safe to call concurrently from multiple threads (torch
        releases the GIL inside `torch.load` for the file read portion)."""
        return self._load_tensor(path)

    # --- internals ------------------------------------------------------------

    def _filename_for(self, uniref_id: str) -> str:
        """Map a UniRef50 id to its on-disk filename.

        The shard files drop the `UniRef50_` prefix (matching how they were
        originally generated). If a caller already passed a bare id, leave
        it alone.
        """
        bare = uniref_id
        if self._id_prefix_strip and bare.startswith(self._id_prefix_strip):
            bare = bare[len(self._id_prefix_strip) :]
        return f"{bare}.pt"

    def _resolve_path(self, filename: str, index_by_filename: dict[str, str]) -> Path | None:
        # 1. SQLite index hit.
        folder = index_by_filename.get(filename)
        if folder is not None:
            candidate = self.root / folder / filename
            if candidate.is_file():
                return candidate
            # Stale row — file moved / deleted. Fall through to fallback dirs.
            logger.warning("shard_store: index row points at missing file %s", candidate)

        # 2. Fallback dirs — flat .pt files, possibly with _F<n> suffixes.
        stem = filename[:-3]  # strip ".pt"
        for d in self._fallback_dirs:
            if not d.is_dir():
                continue
            exact = d / filename
            if exact.is_file():
                return exact
            # Fragment variants: pick the lexicographically-last file so
            # later fragments win. Not scientifically defensible — the
            # producer of these dirs should document it — but deterministic
            # and auditable from the DEBUG log below.
            fragments = sorted(glob.glob(str(d / f"{stem}_F*.pt")))
            if fragments:
                logger.debug(
                    "shard_store: resolved %s via fragment variant %s",
                    filename,
                    fragments[-1],
                )
                return Path(fragments[-1])
        return None

    def _index_lookup_batch(self, filenames: list[str]) -> dict[str, str]:
        """Bulk query `index.db` for a batch of filenames.

        Uses a single parameterised IN-clause to avoid per-id round trips.
        SQLite has a default 999-param limit; we chunk at 500 for headroom.
        """
        out: dict[str, str] = {}
        if not filenames:
            return out
        try:
            # `immutable=1` promises we won't write; sqlite then skips the
            # WAL + locking machinery entirely.
            uri = f"file:{self._index_db}?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        except sqlite3.Error as exc:
            logger.warning("shard_store: sqlite open failed: %s", exc)
            return out
        try:
            conn.execute("PRAGMA query_only=ON")
            for start in range(0, len(filenames), 500):
                chunk = filenames[start : start + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT file_path, folder_name FROM files WHERE file_path IN ({placeholders})",
                    chunk,
                ).fetchall()
                for fp, folder in rows:
                    out[fp] = folder
        except sqlite3.Error as exc:
            logger.warning("shard_store: sqlite query failed: %s", exc)
        finally:
            conn.close()
        return out

    def _load_tensor(self, path: Path) -> np.ndarray | None:
        """Read one `.pt` file and return it as a float32 ndarray.

        `weights_only=True` prevents arbitrary pickle execution; the shards
        are trusted data but there's no reason to leave the door open.
        """
        try:
            import torch  # heavy import; lazy

            tensor = torch.load(str(path), map_location="cpu", weights_only=True)
        except FileNotFoundError:
            logger.warning("shard_store: %s disappeared mid-read", path)
            return None
        except Exception:
            logger.exception("shard_store: torch.load failed for %s", path)
            return None

        arr: np.ndarray = (
            tensor.detach().to(dtype=self._np_dtype()).numpy()
            if hasattr(tensor, "detach")
            else np.asarray(tensor)
        )
        if arr.ndim != 2 or arr.shape[-1] != self.dim:
            raise ShardDimMismatch(
                f"shard_store: {path} has shape {arr.shape}, expected (*, {self.dim})"
            )
        return arr

    @staticmethod
    def _np_dtype() -> Any:
        # Imported lazily so ShardStore's module import doesn't pull torch.
        import torch

        return torch.float32


__all__ = ["ShardDimMismatch", "ShardDimMismatchError", "ShardStore"]
