"""A3M header surgery for templates re-alignment.

Two operations on the existing `>id/start-end <description>` header:

1. `reinterval_header(header, new_start, new_end)` — replace `/start-end`
   with the realignment's actual span. Domain id and every tail token
   (`[subseq from]`, `mol:protein`, `length:720`, `Exostosin-1`, ...)
   are preserved verbatim.

2. `stamp_score(header, score)` — strip any prior `Score=...` token,
   append `Score={score:.3f}` at the end.

These are pure string transforms — no parsing, no validation. The
parser already decided the header is well-formed (records with a
malformed header land in `result.dropped`), so these helpers stay
defensive only at the edges.
"""

from __future__ import annotations

import re
from typing import Final

# Match the leading `>id/start-end` token. `\S+?` is non-greedy: it
# expands via backtracking until `/digits-digits` lines up immediately
# before whitespace or end-of-string. This means ids that themselves
# contain forward slashes (e.g. `>foo/bar/123-456`) parse correctly —
# `\S+?` backtracks all the way to `>foo/bar` so the `/123-456` part
# becomes the interval.
_HEADER_TOKEN_RE: Final = re.compile(r"\A(>\S+?)/(\d+)-(\d+)(?=\s|\Z)")

# Score=/score=/Score:/score: (case-insensitive, both separators).
# Stripped before stamping. Leading whitespace requirement avoids
# accidentally eating an id that begins with `Score=...`.
_SCORE_TOKEN_RE: Final = re.compile(r"\s+[Ss]core[:=]\S+")


def reinterval_header(header: str, new_start: int, new_end: int) -> str:
    """Replace the `/start-end` span in `header` with `new_start-new_end`.

    Header layout: `>id/start-end<sep><tail>` where `<sep>` is whitespace
    (or end-of-string for headers with no tail). The id and tail are
    preserved verbatim. Headers without a recognizable `/start-end`
    token are returned unchanged — defensive fallback only; the parser
    drops malformed records before this is called.
    """
    m = _HEADER_TOKEN_RE.match(header)
    if m is None:
        return header
    return f"{m.group(1)}/{new_start}-{new_end}{header[m.end():]}"


def stamp_score(header: str, score: float) -> str:
    """Strip any pre-existing `Score=...` / `score=...` / `Score:...` /
    `score:...` token from `header`, then insert `score:N.NNN` at the
    end of the technical-tokens section.

    hmmsearch headers follow the convention
        `>id/start-end <key:value tokens>  <free-text description>`
    where a **double-space gap** separates technical tokens (like
    `mol:protein`, `length:720`) from the human-readable description
    (like `Exostosin-1`). We slot `score:N.NNN` in next to those
    technical tokens — right before the double-space — so it lives
    where downstream parsers expect machine-readable fields. Headers
    without a double-space separator get `score:N.NNN` appended at
    the end.

    Multiple existing score tokens (rare but possible if a caller
    stamped twice) are all stripped before the new one is inserted.
    """
    cleaned = _SCORE_TOKEN_RE.sub("", header).rstrip()
    score_token = f"score:{score:.3f}"

    # Collapse any double-double-spaces that the score-strip may have
    # left behind (e.g. `... length:720  score:0.5  Exostosin-1` →
    # stripping middle leaves `... length:720   Exostosin-1`).
    while "   " in cleaned:
        cleaned = cleaned.replace("   ", "  ")

    sep_idx = cleaned.find("  ")
    if sep_idx >= 0:
        return f"{cleaned[:sep_idx]} {score_token}{cleaned[sep_idx:]}"
    return f"{cleaned} {score_token}"


__all__ = ["reinterval_header", "stamp_score"]
