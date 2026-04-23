"""Populate a Redis path index from the ShardStore sqlite index.

Mirrors upstream DeepFold-PLM's design: each shard filename maps to its
parent directory via a Redis key, so the embedding service can resolve
paths with one `MGET` instead of four chunked `SELECT ... IN (?,?,...)`
queries against a `/gpfs`-hosted sqlite file (measured at ~8 ms/row,
~13 s/request for 1500 ids).

Key format
----------
``shard:<model>:<bare_id>`` → ``<folder_name>``

Where ``<bare_id>`` matches the shard filename stem (no ``UniRef50_``
prefix, no ``.pt`` extension). Default ``<model>=prott5`` for the
ProtT5 shards.

Usage
-----
::

    python -m plmmsa.tools.build_shard_index \\
        --sqlite /gpfs/database/milvus/datasets/uniref50_t5/datasets/index.db \\
        --redis-url redis://localhost:6379/0 \\
        --model prott5 \\
        --batch 10000

Idempotent — re-running updates the same keys in-place. To clear only
the shard index (leaves ``seq:*`` / ``tax:*`` alone)::

    redis-cli --scan --pattern 'shard:prott5:*' | xargs -n 500 redis-cli del
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from pathlib import Path

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


def iter_sqlite_rows(db_path: Path) -> list[tuple[str, str]]:
    """Stream `(filename, folder)` rows from the shard sqlite index.

    The index is read-only so we can open it immutable — no lock, no
    WAL writes, no journal creation.
    """
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute("SELECT file_path, folder_name FROM files")
        return cur.fetchall()
    finally:
        conn.close()


async def populate(
    *,
    sqlite_path: Path,
    redis_url: str,
    model: str,
    batch: int,
    key_prefix: str,
) -> int:
    """Pump sqlite rows into Redis via pipelined SET. Returns row count."""
    client = Redis.from_url(redis_url, decode_responses=False)
    try:
        rows = await asyncio.to_thread(iter_sqlite_rows, sqlite_path)
        total = len(rows)
        logger.info("shard_index: %d rows in %s", total, sqlite_path)

        # Chunk pipeline to avoid single giant MULTI block. 10k is a
        # good balance between round-trips and per-pipeline memory
        # (~200 KB at 20-byte keys).
        prefix = key_prefix or f"shard:{model}:"
        written = 0
        for start in range(0, total, batch):
            chunk = rows[start : start + batch]
            pipe = client.pipeline(transaction=False)
            for filename, folder in chunk:
                # Strip `.pt` suffix: index keys are by filename, Redis
                # keys are by bare id to save space and make
                # client-side key construction trivial.
                bare = filename[:-3] if filename.endswith(".pt") else filename
                pipe.set(f"{prefix}{bare}", str(folder).encode())
            await pipe.execute()
            written += len(chunk)
            if written % (batch * 10) == 0 or written == total:
                logger.info("shard_index: %d / %d written", written, total)
        return written
    finally:
        await client.aclose()


def build_argparser() -> argparse.ArgumentParser:
    desc = (__doc__ or "").splitlines()[0]
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument(
        "--sqlite",
        required=True,
        type=Path,
        help="Path to the ShardStore sqlite index (e.g. /gpfs/.../uniref50_t5/datasets/index.db).",
    )
    ap.add_argument(
        "--redis-url",
        required=True,
        help="Target Redis URL (e.g. redis://cache-seq:6379/0).",
    )
    ap.add_argument(
        "--model",
        default="prott5",
        help="Model id — shapes the key prefix `shard:<model>:`.",
    )
    ap.add_argument(
        "--key-prefix",
        default="",
        help="Override the full key prefix (default `shard:<model>:`). "
        "Set this only if you're mirroring a custom layout.",
    )
    ap.add_argument(
        "--batch",
        default=10_000,
        type=int,
        help="Pipeline batch size. 10k is a good balance; raise on a "
        "local socket, lower over a slower link.",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (DEBUG / INFO / WARNING).",
    )
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.sqlite.is_file():
        logger.error("shard_index: sqlite not found at %s", args.sqlite)
        return 2

    written = asyncio.run(
        populate(
            sqlite_path=args.sqlite,
            redis_url=args.redis_url,
            model=args.model,
            batch=args.batch,
            key_prefix=args.key_prefix,
        )
    )
    logger.info("shard_index: done, %d keys written", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
