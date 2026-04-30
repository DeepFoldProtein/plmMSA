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

# Score= or score= (case-insensitive). Stripped before stamping. Leading
# whitespace requirement avoids accidentally eating an id that begins
# with `Score=...` (which would be a malformed header anyway).
_SCORE_TOKEN_RE: Final = re.compile(r"\s+[Ss]core=\S+")


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
    """Strip any pre-existing `Score=...` (or `score=...`) token from
    `header`, then insert the new `Score=...` right after the first
    whitespace-separated header token (typically `>id/start-end`).

    Layout: `>id/start-end Score=N.NNN <description tail>`. Putting
    Score= adjacent to the id/range — instead of trailing the
    description — keeps the alignment metric next to the locus it
    describes and makes the header diff-friendly (the score moves but
    the description doesn't shift columns).

    Multiple existing Score= tokens (rare but possible if a caller
    stamped twice) all get stripped. Headers with no description tail
    just get `>id/start-end Score=N.NNN`.
    """
    cleaned = _SCORE_TOKEN_RE.sub("", header).rstrip()
    score_token = f"Score={score:.3f}"
    parts = cleaned.split(maxsplit=1)
    if len(parts) == 1:
        return f"{parts[0]} {score_token}"
    return f"{parts[0]} {score_token} {parts[1]}"


__all__ = ["reinterval_header", "stamp_score"]
