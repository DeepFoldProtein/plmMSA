"""A3M row renderer for the templates re-alignment pipeline.

Distinct from `plmmsa.pipeline.a3m.render_hit`: that helper preserves
lowercase insertions (template residues without a query column,
`(qi=-1, ti>=0)` columns). This module's `render_hit_match_only` drops
them — the pipeline rule is "no lowercase in the output row" (PLAN §2),
so any template residue OTalign couldn't place at a query column is
trimmed from the rendered row.

Output invariants:
  - `len(row) == query_len` always.
  - `set(row) ⊆ set("ACDEFGHIKLMNPQRSTVWY-")` — no lowercase, no other
    characters. (We don't enforce the canonical 20-residue alphabet
    here; we trust the caller's `target_seq`.)
"""

from __future__ import annotations

from collections.abc import Sequence


def render_hit_match_only(
    query_len: int,
    target_seq: str,
    columns: Sequence[tuple[int, int]],
) -> str:
    """Render an A3M row containing only match-state slots.

    `columns` is an OTalign-style alignment path: each `(qi, ti)` is a
    column in the alignment, with `-1` marking a gap on that side. The
    only columns that contribute to the rendered row are
    `(qi>=0, ti>=0)` (match columns), where the template residue at
    index `ti` lands in slot `qi`. Columns with `qi=-1, ti>=0` (template
    residues without a query column) are dropped — the pipeline does
    NOT emit lowercase insertions. Columns with `qi>=0, ti=-1` (gap in
    target) leave the slot at its default `-`.

    Slots not covered by any match column stay as `-`. This guarantees
    the output is exactly `query_len` characters and contains only
    uppercase residues + `-`.
    """
    slots = ["-"] * query_len
    for qi, ti in columns:
        if 0 <= qi < query_len and ti >= 0:
            slots[qi] = target_seq[ti].upper()
    return "".join(slots)


def kept_template_span(
    columns: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """Min and max template residue indices that landed in match
    columns. Returns `None` when no template residue was placed.

    Used to re-interval the header after the no-lowercase filter:
    `new_start = orig_start + kept_lo`, `new_end = orig_start + kept_hi`
    (both 1-based when `orig_start` is). When OTalign drops interior
    template residues, `kept_hi - kept_lo + 1` exceeds the number of
    placed residues — see PLAN §2 for the trade-off.
    """
    lo: int | None = None
    hi: int | None = None
    for qi, ti in columns:
        if qi >= 0 and ti >= 0:
            if lo is None or ti < lo:
                lo = ti
            if hi is None or ti > hi:
                hi = ti
    if lo is None or hi is None:
        return None
    return lo, hi


__all__ = ["kept_template_span", "render_hit_match_only"]
