"""Populate a Redis sequence cache from a FASTA file.

Streams through the FASTA — suitable for multi-GB UniRef inputs that don't fit
in memory. Keys are written using a format string (default ``seq:{id}``) that
must match whatever `RedisTargetFetcher` is configured with.

Usage::

    python -m plmmsa.tools.build_sequence_cache \\
        --fasta /path/to/uniref50.fasta \\
        --redis-url redis://localhost:6379 \\
        --collection ankh_uniref50

``--collection`` is optional; it's only interpolated into the key format if
the format string references ``{collection}``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from redis.asyncio import Redis

from plmmsa.pipeline.fetcher import DEFAULT_SEQ_KEY_FORMAT

logger = logging.getLogger(__name__)


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield `(id, sequence)` pairs from a FASTA file.

    The id is the first whitespace-delimited token after ``>``. Multi-line
    sequences are concatenated. Blank lines are skipped.
    """
    current_id: str | None = None
    parts: list[str] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(parts)
                current_id = line[1:].split(maxsplit=1)[0]
                parts = []
            else:
                parts.append(line)
    if current_id is not None:
        yield current_id, "".join(parts)


async def build(
    *,
    fasta_path: Path,
    redis: Redis | None = None,
    redis_url: str | None = None,
    key_format: str = DEFAULT_SEQ_KEY_FORMAT,
    collection: str = "",
    batch_size: int = 1000,
) -> int:
    """Populate Redis with `{id → sequence}` from the FASTA. Returns count written."""
    owns_redis = False
    if redis is None:
        if redis_url is None:
            raise ValueError("either `redis` or `redis_url` must be provided")
        redis = Redis.from_url(redis_url, decode_responses=False)
        owns_redis = True

    count = 0
    try:
        pipeline = redis.pipeline(transaction=False)
        batched = 0
        for seq_id, seq in iter_fasta(fasta_path):
            key = key_format.format(id=seq_id, collection=collection)
            pipeline.set(key, seq.encode("utf-8"))
            batched += 1
            if batched >= batch_size:
                await pipeline.execute()
                count += batched
                batched = 0
                pipeline = redis.pipeline(transaction=False)
                if count % (batch_size * 10) == 0:
                    logger.info("build_sequence_cache: %d keys written so far", count)
        if batched > 0:
            await pipeline.execute()
            count += batched
    finally:
        if owns_redis:
            await redis.aclose()

    logger.info("build_sequence_cache: wrote %d keys from %s", count, fasta_path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate a Redis sequence cache from a FASTA file.",
    )
    parser.add_argument("--fasta", type=Path, required=True, help="FASTA file to load.")
    parser.add_argument(
        "--redis-url", type=str, required=True, help="Redis URL (e.g. redis://cache:6379)."
    )
    parser.add_argument(
        "--key-format",
        type=str,
        default=DEFAULT_SEQ_KEY_FORMAT,
        help=f"Redis key format (default: {DEFAULT_SEQ_KEY_FORMAT!r}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Interpolated into key-format if it references {collection}.",
    )
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = asyncio.run(
        build(
            fasta_path=args.fasta,
            redis_url=args.redis_url,
            key_format=args.key_format,
            collection=args.collection,
            batch_size=args.batch,
        )
    )
    print(f"Wrote {count} keys to {args.redis_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
