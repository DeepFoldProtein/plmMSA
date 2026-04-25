"""Populate a Redis sequence cache from a FASTA file or a UniRef50 CSV split.

Streams through the input — suitable for multi-GB UniRef inputs that don't fit
in memory. Emits two key spaces:

- ``seq:{id}``  → sequence bytes (utf-8).
- ``tax:{id}``  → NCBI TaxID as bytes (utf-8) — only written when the source
  carries a ``TaxID=<n>`` field (e.g. UniRef50 records). Use ``--no-tax``
  to suppress, or ``--tax-key-format ""``.

``--collection`` is interpolated into the key format if it references
``{collection}``.

Usage (FASTA)::

    python -m plmmsa.tools.build_sequence_cache \\
        --fasta /path/to/uniref50.fasta \\
        --redis-url redis://localhost:6379

Usage (UniRef50 CSV split, e.g. the ``uniref50_t5/split`` layout)::

    python -m plmmsa.tools.build_sequence_cache \\
        --csv-dir /gpfs/database/milvus/datasets/uniref50_t5/split \\
        --redis-url redis://localhost:6379 \\
        --key-format 'seq:UniRef50_{id}' \\
        --tax-key-format 'tax:UniRef50_{id}'

The CSV schema expected by ``--csv`` / ``--csv-dir`` is
``accession, description, sequence, length, length_group`` (extra columns are
ignored). The UniRef50 cluster id and ``TaxID=<n>`` both live in
``description``, matching the FASTA header format; we reuse the same regex.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
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


def iter_csv(path: Path) -> Iterator[tuple[str, str, str | None]]:
    """Yield `(id, sequence, tax_id_or_None)` triples from one UniRef50 CSV.

    Expects columns ``accession, description, sequence`` (the full schema
    also has ``length, length_group`` — ignored). TaxID comes from the
    ``description`` column via the same ``TaxID=<n>`` regex used for FASTA
    headers.
    """
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if not acc or not seq:
                continue
            yield acc, seq, _extract_tax_id(row.get("description"))


def iter_csv_dir(
    root: Path, *, start: int = 0, stop: int | None = None
) -> Iterator[tuple[str, str, str | None]]:
    """Yield triples from every `*.csv` under `root`, lex-sorted.

    `start` / `stop` slice the sorted file list so an interrupted run can
    resume at a known file index. Each file boundary is logged at INFO so
    the operator can pick a sensible `--start` after a crash.
    """
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no *.csv files under {root}")
    selected = files[start:stop]
    total = len(selected)
    logger.info(
        "csv-dir: %d files selected (of %d total) from %s (start=%d, stop=%s)",
        total,
        len(files),
        root,
        start,
        "end" if stop is None else stop,
    )
    for idx, path in enumerate(selected):
        global_idx = start + idx
        logger.info("csv-dir: [%d/%d] %s", global_idx, start + total - 1, path.name)
        yield from iter_csv(path)


def _extract_tax_id(header: str | None) -> str | None:
    if header is None:
        return None
    match = _TAXID_RE.search(header)
    return match.group(1) if match else None


async def build(
    *,
    records: Iterator[tuple[str, str, str | None]],
    redis: Redis | None = None,
    redis_url: str | None = None,
    key_format: str = DEFAULT_SEQ_KEY_FORMAT,
    tax_key_format: str | None = DEFAULT_TAX_KEY_FORMAT,
    collection: str = "",
    batch_size: int = 1000,
) -> tuple[int, int]:
    """Drain `records` into Redis as `{id → sequence}` + `{id → tax_id}`.

    Accepts any iterator of `(id, sequence, tax_id_or_None)` — the FASTA
    and CSV front-ends both produce this shape. Returns `(seq_count,
    tax_count)`.
    """
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
        for seq_id, seq, tax_id in records:
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
        "build_sequence_cache: wrote %d seq keys and %d tax keys",
        seq_count,
        tax_count,
    )
    return seq_count, tax_count


# Kept for backwards compatibility with existing callers that pass a
# FASTA path directly. Thin shim over `build()`.
async def build_from_fasta(
    *,
    fasta_path: Path,
    redis: Redis | None = None,
    redis_url: str | None = None,
    key_format: str = DEFAULT_SEQ_KEY_FORMAT,
    tax_key_format: str | None = DEFAULT_TAX_KEY_FORMAT,
    collection: str = "",
    batch_size: int = 1000,
) -> tuple[int, int]:
    return await build(
        records=iter_fasta(fasta_path),
        redis=redis,
        redis_url=redis_url,
        key_format=key_format,
        tax_key_format=tax_key_format,
        collection=collection,
        batch_size=batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Populate a Redis sequence cache from a FASTA file or UniRef50 CSV split.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=Path, help="FASTA file to load.")
    source.add_argument(
        "--csv",
        type=Path,
        dest="csv_file",
        help="Single UniRef50 CSV file (columns: accession, description, sequence, ...).",
    )
    source.add_argument(
        "--csv-dir",
        type=Path,
        help=(
            "Directory of UniRef50 CSV shards (same schema as --csv). "
            "Files are discovered via `*.csv` and lex-sorted; use --start / --stop "
            "to resume an interrupted run."
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="csv-dir only: index of the first file in the sorted glob to process (inclusive).",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="csv-dir only: index to stop before (exclusive). Omit to go to the end.",
    )
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

    if args.fasta is not None:
        records: Iterator[tuple[str, str, str | None]] = iter_fasta(args.fasta)
        source_desc = str(args.fasta)
    elif args.csv_file is not None:
        records = iter_csv(args.csv_file)
        source_desc = str(args.csv_file)
    else:
        records = iter_csv_dir(args.csv_dir, start=args.start, stop=args.stop)
        source_desc = f"{args.csv_dir} [start={args.start} stop={args.stop}]"

    seq_count, tax_count = asyncio.run(
        build(
            records=records,
            redis_url=args.redis_url,
            key_format=args.key_format,
            tax_key_format=tax_fmt,
            collection=args.collection,
            batch_size=args.batch,
        )
    )
    print(f"Wrote {seq_count} seq keys, {tax_count} tax keys to {args.redis_url} (source: {source_desc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build",
    "build_from_fasta",
    "iter_csv",
    "iter_csv_dir",
    "iter_fasta",
    "main",
]
