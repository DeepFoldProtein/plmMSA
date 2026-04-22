"""Populate a Redis sequence cache from a FASTA file.

Streams through the FASTA — suitable for multi-GB UniRef inputs that don't fit
in memory. Emits two key spaces:

- ``seq:{id}``  → sequence bytes (utf-8).
- ``tax:{id}``  → NCBI TaxID as bytes (utf-8) — only written when the header
  contains a ``TaxID=<n>`` field (e.g. UniRef50 records). Use ``--no-tax``
  to suppress, or ``--tax-key-format ""``.

``--collection`` is interpolated into the key format if it references
``{collection}``.

Usage::

    python -m plmmsa.tools.build_sequence_cache \\
        --fasta /path/to/uniref50.fasta \\
        --redis-url redis://localhost:6379
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from redis.asyncio import Redis

from plmmsa.pipeline.fetcher import DEFAULT_SEQ_KEY_FORMAT

DEFAULT_TAX_KEY_FORMAT = "tax:{id}"
_TAXID_RE = re.compile(r"\bTaxID=(\S+)")

logger = logging.getLogger(__name__)


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str | None]]:
    """Yield `(id, sequence, tax_id_or_None)` triples from a FASTA file.

    The id is the first whitespace-delimited token after ``>``. Multi-line
    sequences are concatenated. Blank lines are skipped. `tax_id` is the
    `TaxID=` field from the header if present (UniRef-style), else None.
    """
    current_id: str | None = None
    current_header: str | None = None
    parts: list[str] = []
    with path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(parts), _extract_tax_id(current_header)
                header = line[1:]
                current_id = header.split(maxsplit=1)[0]
                current_header = header
                parts = []
            else:
                parts.append(line)
    if current_id is not None:
        yield current_id, "".join(parts), _extract_tax_id(current_header)


def _extract_tax_id(header: str | None) -> str | None:
    if header is None:
        return None
    match = _TAXID_RE.search(header)
    return match.group(1) if match else None


async def build(
    *,
    fasta_path: Path,
    redis: Redis | None = None,
    redis_url: str | None = None,
    key_format: str = DEFAULT_SEQ_KEY_FORMAT,
    tax_key_format: str | None = DEFAULT_TAX_KEY_FORMAT,
    collection: str = "",
    batch_size: int = 1000,
) -> tuple[int, int]:
    """Populate Redis with `{id → sequence}` and (optionally) `{id → tax_id}`
    from the FASTA. Returns `(seq_count, tax_count)`."""
    owns_redis = False
    if redis is None:
        if redis_url is None:
            raise ValueError("either `redis` or `redis_url` must be provided")
        redis = Redis.from_url(redis_url, decode_responses=False)
        owns_redis = True

    seq_count = 0
    tax_count = 0
    try:
        pipeline = redis.pipeline(transaction=False)
        batched_writes = 0  # pipeline.set() calls since last execute()
        last_log = 0
        for seq_id, seq, tax_id in iter_fasta(fasta_path):
            seq_key = key_format.format(id=seq_id, collection=collection)
            pipeline.set(seq_key, seq.encode("utf-8"))
            batched_writes += 1
            seq_count += 1
            if tax_key_format and tax_id is not None:
                tax_key = tax_key_format.format(id=seq_id, collection=collection)
                pipeline.set(tax_key, tax_id.encode("utf-8"))
                batched_writes += 1
                tax_count += 1
            if batched_writes >= batch_size:
                await pipeline.execute()
                batched_writes = 0
                pipeline = redis.pipeline(transaction=False)
                if seq_count - last_log >= batch_size * 10:
                    logger.info(
                        "build_sequence_cache: %d seq, %d tax written so far",
                        seq_count,
                        tax_count,
                    )
                    last_log = seq_count
        if batched_writes > 0:
            await pipeline.execute()
    finally:
        if owns_redis:
            await redis.aclose()

    logger.info(
        "build_sequence_cache: wrote %d seq keys and %d tax keys from %s",
        seq_count,
        tax_count,
        fasta_path,
    )
    return seq_count, tax_count


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
        help=f"Sequence key format (default: {DEFAULT_SEQ_KEY_FORMAT!r}).",
    )
    parser.add_argument(
        "--tax-key-format",
        type=str,
        default=DEFAULT_TAX_KEY_FORMAT,
        help=(
            f"Taxonomy key format (default: {DEFAULT_TAX_KEY_FORMAT!r}). "
            "Empty string disables tax emission."
        ),
    )
    parser.add_argument(
        "--no-tax",
        action="store_true",
        help="Alias for --tax-key-format=''.",
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

    tax_fmt: str | None = args.tax_key_format if args.tax_key_format and not args.no_tax else None

    seq_count, tax_count = asyncio.run(
        build(
            fasta_path=args.fasta,
            redis_url=args.redis_url,
            key_format=args.key_format,
            tax_key_format=tax_fmt,
            collection=args.collection,
            batch_size=args.batch,
        )
    )
    print(f"Wrote {seq_count} seq keys, {tax_count} tax keys to {args.redis_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
