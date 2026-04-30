"""hmmsearch-style A3M parser for the templates re-alignment pipeline.

Per `PLAN_TEMPLATES_REALIGN.md` §1.2, every record in a hmmsearch A3M
satisfies:

1. `upper + gap == query_len` — match-state column count, constant across
   all records (it's the file-level invariant).
2. `upper + lower == end - start + 1` — non-gap residues equal the header's
   interval length (the actual template residue count).
3. alphabet `[A-Z | a-z | -]` only.

This module parses input text into validated `Record`s. Records that fail
any rule are placed into `result.dropped` with a human-readable reason; we
never raise — the caller decides whether `len(dropped) > 0` is a hard
error or just a stat.

Companion helper `columns_from_a3m_row` derives the OTalign-style column
list from an A3M row — used by the round-trip test in §6.4 and by any
caller that wants to re-render an existing alignment.

Hot loops (per-row character classification + column derivation) are
numba-jitted on uint8 byte arrays. ~6x speedup on the bundled fixture
(33 ms → 5 ms parse, 25 ms → 5 ms columns). First call pays a
one-time compile cost (~1 s); cached to `__pycache__` thereafter.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

import numba
import numpy as np

# Match-state slots are uppercase residues OR `-`. Lowercase letters are
# template-side insertions (no query column). Anything else is rejected.
_HEADER_TOKEN_RE: Final = re.compile(r"\A(.+)/(\d+)-(\d+)\Z")

# Byte values used by the JIT loops. ord() at module load — numba sees
# them as compile-time constants.
_DASH: Final = ord("-")  # 45
_A_UPPER: Final = ord("A")  # 65
_Z_UPPER: Final = ord("Z")  # 90
_A_LOWER: Final = ord("a")  # 97
_Z_LOWER: Final = ord("z")  # 122

# JIT status codes shared by the helpers below.
_STATUS_OK = 0
_STATUS_ALPHABET = 1
_STATUS_QLEN_MISMATCH = 2


@numba.njit(cache=True, nogil=True, fastmath=True)
def _row_stats_jit(row: np.ndarray) -> tuple[int, int, int, int]:
    """Single-pass character classification — counts upper / lower / gap
    and validates the alphabet in one loop.

    Returns `(upper, lower, gap, status)`. `status == 0` means clean;
    `status == 1` means an alphabet violation was hit (counts are
    undefined in that case — caller falls back to a pure-Python pass
    to find the offending character for a useful error message).
    """
    upper = 0
    lower = 0
    gap = 0
    for i in range(len(row)):
        b = row[i]
        if b == _DASH:
            gap += 1
        elif _A_UPPER <= b <= _Z_UPPER:
            upper += 1
        elif _A_LOWER <= b <= _Z_LOWER:
            lower += 1
        else:
            return 0, 0, 0, _STATUS_ALPHABET
    return upper, lower, gap, _STATUS_OK


@numba.njit(cache=True, nogil=True, fastmath=True)
def _columns_from_row_jit(
    row: np.ndarray, query_len: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Derive `(qi, ti)` columns from an A3M row.

    Returns `(qi_arr, ti_arr, status)` — both arrays are int32 of length
    `len(row)`. `status` is one of:
      0 — clean
      1 — alphabet violation (arrays are partially populated; ignore)
      2 — query-length mismatch (arrays are valid; caller can read
          `qi_arr[-1] + 1` for the actual match-state count)
    """
    n = row.shape[0]
    qi_arr = np.empty(n, dtype=np.int32)
    ti_arr = np.empty(n, dtype=np.int32)
    qi = np.int32(0)
    ti = np.int32(0)
    for i in range(n):
        b = row[i]
        if b == _DASH:
            qi_arr[i] = qi
            ti_arr[i] = -1
            qi += 1
        elif _A_UPPER <= b <= _Z_UPPER:
            qi_arr[i] = qi
            ti_arr[i] = ti
            qi += 1
            ti += 1
        elif _A_LOWER <= b <= _Z_LOWER:
            qi_arr[i] = -1
            ti_arr[i] = ti
            ti += 1
        else:
            return qi_arr, ti_arr, _STATUS_ALPHABET
    if int(qi) != query_len:
        return qi_arr, ti_arr, _STATUS_QLEN_MISMATCH
    return qi_arr, ti_arr, _STATUS_OK


def _row_to_bytes(row: str) -> np.ndarray:
    """ASCII string → uint8 ndarray (zero-copy view via frombuffer)."""
    return np.frombuffer(row.encode("ascii"), dtype=np.uint8)


@dataclass(slots=True)
class Record:
    """One validated A3M record.

    `header` is the original `>...` line preserved verbatim — we only
    edit it at render time (re-interval `/start-end` and stamp `Score=`,
    per PLAN §2). `description` is everything after the first
    whitespace-separated header token; we keep it so domain names + tail
    metadata round-trip unchanged through the pipeline (`Exostosin-1`,
    `[subseq from]`, `mol:protein`, `length:720`, ...).
    """

    header: str
    target_id: str  # e.g. "7sch_A"
    start: int  # 1-based, inclusive
    end: int  # 1-based, inclusive
    row: str  # full A3M row, verbatim (mixed case + `-`)
    raw_seq: str  # uppercase, gap-free residues — what we feed to the PLM
    description: str = ""  # tail of the header after `id/start-end`


@dataclass(slots=True)
class DroppedRecord:
    """A record that failed sanity checks. `reason` is a stable short
    code so the caller can aggregate by type (used by §6.3's stats and
    by the orchestrator's `records_dropped_*` counters)."""

    header: str
    reason: str


@dataclass(slots=True)
class ParseResult:
    """Output of `parse_hmmsearch_a3m`.

    `query_len` is the file-level match-state count (constant across all
    surviving records). `records` is the list of records that passed all
    sanity checks; `dropped` collects the rest with a reason each.
    """

    query_len: int
    records: list[Record] = field(default_factory=list)
    dropped: list[DroppedRecord] = field(default_factory=list)


def columns_from_a3m_row(row: str, query_len: int) -> list[tuple[int, int]]:
    """Derive the OTalign-style column list from an A3M row.

    Conventions:
      - uppercase letter at position k → match-state column. `(qi, ti)`,
        both indices increment.
      - "-" at position k → gap in the target at this query column.
        `(qi, -1)`, only `qi` increments.
      - lowercase letter → insertion in the template, no query column.
        `(-1, ti)`, only `ti` increments.

    Round-trip property: feeding the returned column list (along with the
    record's `raw_seq`) into `plmmsa.pipeline.a3m.render_hit` reproduces
    the original row byte-for-byte. Pinned in §6.4.

    Raises `ValueError` on alphabet violations or query-length mismatch.
    """
    qi_arr, ti_arr, status = _columns_from_row_jit(_row_to_bytes(row), query_len)
    if status == _STATUS_ALPHABET:
        # Re-find the offender in Python so the error message is useful.
        for c in row:
            if not (c == "-" or c.isupper() or c.islower()):
                raise ValueError(f"unexpected character {c!r} in A3M row")
        raise ValueError("unexpected character in A3M row")
    if status == _STATUS_QLEN_MISMATCH:
        match_cols = sum(1 for c in row if c == "-" or c.isupper())
        raise ValueError(
            f"row has {match_cols} match-state slots, expected {query_len}"
        )
    # Materialize into list-of-tuples — this is the API contract
    # (render_hit iterates `for qi, ti in hit.columns`). zip on plain
    # ints (via `tolist()`) is faster than tuple-by-tuple indexing.
    return list(zip(qi_arr.tolist(), ti_arr.tolist(), strict=True))


def _split_header(line: str) -> tuple[str, str, int, int] | None:
    """Parse `>id/start-end <description>`.

    Returns `(target_id, description, start, end)` on success, `None`
    on a malformed header. We only validate the *first* whitespace-
    separated token; everything after is kept verbatim as
    `description` (see `Record.description`).
    """
    if not line.startswith(">"):
        return None
    body = line[1:].strip()
    if not body:
        return None
    parts = body.split(maxsplit=1)
    first = parts[0]
    m = _HEADER_TOKEN_RE.match(first)
    if not m:
        return None
    target_id, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    if e < s:
        return None
    description = parts[1] if len(parts) > 1 else ""
    return target_id, description, s, e


def _iter_records(text: str) -> Iterator[tuple[str, str]]:
    """Yield `(header_line, body_text)` pairs.

    Handles multi-line records (sequence split across multiple lines),
    blank lines between records, and `#` comments at line start.
    """
    cur_header: str | None = None
    cur_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if cur_header is not None:
                yield cur_header, "".join(cur_lines)
            cur_header = line
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_header is not None:
        yield cur_header, "".join(cur_lines)


def parse_hmmsearch_a3m(
    text: str,
    *,
    expected_query_len: int | None = None,
) -> ParseResult:
    """Parse a hmmsearch-style A3M text into validated records.

    Rules from PLAN §1.2:
      - `upper + gap == query_len` (constant across the file).
      - `upper + lower == end - start + 1` (interval matches residue count).
      - alphabet is `[A-Za-z-]` only.

    `expected_query_len`:
      - `None` (default): derive from the first sane record's
        `upper + gap` count. Subsequent records that disagree are
        dropped with `query_len_mismatch:<got>vs<expected>`.
      - explicit int: every record must match. Useful when the caller
        already knows the query length (typically because they have the
        normalized query sequence in hand) and wants the parser to fail
        loudly on disagreement.

    Returns the populated `ParseResult`. We never raise — empty input
    yields `ParseResult(query_len=0, records=[], dropped=[])`.
    """
    records: list[Record] = []
    dropped: list[DroppedRecord] = []
    query_len: int | None = expected_query_len

    for header, row in _iter_records(text):
        parsed = _split_header(header)
        if parsed is None:
            dropped.append(DroppedRecord(header, "malformed_header"))
            continue
        target_id, description, start, end = parsed

        upper, lower, gap, status = _row_stats_jit(_row_to_bytes(row))
        if status == _STATUS_ALPHABET:
            dropped.append(DroppedRecord(header, "alphabet_error"))
            continue

        match_cols = upper + gap
        nongap = upper + lower
        interval = end - start + 1

        if nongap != interval:
            dropped.append(
                DroppedRecord(header, f"interval_mismatch:{nongap}vs{interval}")
            )
            continue

        if query_len is None:
            query_len = match_cols
        elif match_cols != query_len:
            dropped.append(
                DroppedRecord(header, f"query_len_mismatch:{match_cols}vs{query_len}")
            )
            continue

        # `replace` + `upper` are C-level — much faster than a Python
        # comprehension. raw_seq is the residue stream we feed to the PLM.
        raw_seq = row.replace("-", "").upper()
        records.append(
            Record(
                header=header,
                target_id=target_id,
                start=start,
                end=end,
                row=row,
                raw_seq=raw_seq,
                description=description,
            )
        )

    return ParseResult(
        query_len=query_len or 0,
        records=records,
        dropped=dropped,
    )


__all__ = [
    "DroppedRecord",
    "ParseResult",
    "Record",
    "columns_from_a3m_row",
    "parse_hmmsearch_a3m",
]
