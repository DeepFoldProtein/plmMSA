"""Pure-function unit tests for templates renderer + header helpers.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.5. Three building blocks the
orchestrator will compose:

- `render_hit_match_only` — produce an A3M row with no lowercase, exactly
  `query_len` chars from `[A-Z-]`.
- `kept_template_span` — recover `(min_ti, max_ti)` of placed template
  residues, used to re-interval the header.
- `reinterval_header` / `stamp_score` — surgical edits on the original
  header that preserve the domain name + all tail tokens.

These are tested standalone (no orchestrator, no model, no httpx) so
that the §6.5 stubbed-pipeline tests can assume the helpers behave.
"""

from __future__ import annotations

import pytest

from plmmsa.templates import (
    kept_template_span,
    reinterval_header,
    render_hit_match_only,
    stamp_score,
)


# ---------------------------------------------------------------------------
# render_hit_match_only
# ---------------------------------------------------------------------------


def test_render_pure_match_alignment() -> None:
    """Identity-style alignment: every template residue lines up with a
    distinct query column → row is the template residues at those slots,
    nothing else."""
    cols = [(0, 0), (1, 1), (2, 2)]
    row = render_hit_match_only(query_len=5, target_seq="ACD", columns=cols)
    assert row == "ACD--"


def test_render_strips_lowercase_inserts() -> None:
    """Columns with `qi=-1, ti>=0` (template insertions) are dropped —
    no lowercase in the output. Slots that do have a match column show
    the residue uppercase regardless of `target_seq` casing."""
    cols = [(0, 0), (-1, 1), (2, 2), (-1, 3)]
    row = render_hit_match_only(query_len=4, target_seq="aCdE", columns=cols)
    assert row == "A-D-"
    # Output is uppercase + `-` only.
    assert all(c == "-" or c.isupper() for c in row)


def test_render_gap_pads_uncovered_query_positions() -> None:
    """Synthetic OTalign output covering only `q ∈ [10, 50)` of a
    100-residue query → first 10 and last 50 slots are `-`, middle 40
    are uppercase residues. PLAN §6.5 invariant."""
    cols = [(q, q - 10) for q in range(10, 50)]
    target = "X" * 40
    row = render_hit_match_only(query_len=100, target_seq=target, columns=cols)
    assert len(row) == 100
    assert row.count("-") == 60
    assert row.startswith("-" * 10)
    assert row.endswith("-" * 50)
    assert row[10:50] == "X" * 40


def test_render_gap_in_target_columns_leave_dash() -> None:
    """`(qi>=0, ti=-1)` columns (gap in target at this query position)
    leave the slot at its default `-`."""
    cols = [(0, 0), (1, -1), (2, 1)]
    row = render_hit_match_only(query_len=3, target_seq="AC", columns=cols)
    assert row == "A-C"


def test_render_out_of_range_qi_silently_dropped() -> None:
    """A column with `qi >= query_len` is ignored — defensive against an
    upstream bug rather than a pin on caller behavior."""
    cols = [(0, 0), (5, 1), (1, 2)]  # qi=5 is out of range for query_len=3
    row = render_hit_match_only(query_len=3, target_seq="ACD", columns=cols)
    assert row == "AD-"


def test_render_empty_alignment() -> None:
    row = render_hit_match_only(query_len=5, target_seq="", columns=[])
    assert row == "-----"


# ---------------------------------------------------------------------------
# kept_template_span
# ---------------------------------------------------------------------------


def test_kept_span_contiguous() -> None:
    cols = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert kept_template_span(cols) == (0, 3)


def test_kept_span_with_interior_drops() -> None:
    """Insertions don't change the span — kept_template_span only looks
    at `(qi>=0, ti>=0)` columns. If template indices 0, 1, 5 land in
    match columns and 2, 3, 4 are insertions, the span is (0, 5)."""
    cols = [
        (0, 0),
        (1, 1),
        (-1, 2),  # interior insert (dropped)
        (-1, 3),
        (-1, 4),
        (2, 5),
    ]
    assert kept_template_span(cols) == (0, 5)


def test_kept_span_returns_none_when_no_match_columns() -> None:
    cols = [(0, -1), (1, -1), (-1, 0), (-1, 1)]
    assert kept_template_span(cols) is None


# ---------------------------------------------------------------------------
# reinterval_header
# ---------------------------------------------------------------------------


def test_reinterval_basic() -> None:
    header = ">7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1"
    out = reinterval_header(header, new_start=55, new_end=680)
    assert out == ">7sch_A/55-680 [subseq from] mol:protein length:720  Exostosin-1"


def test_reinterval_preserves_no_tail_case() -> None:
    """Header with no description tail still re-intervals correctly."""
    out = reinterval_header(">A/10-20", new_start=11, new_end=15)
    assert out == ">A/11-15"


def test_reinterval_preserves_id_with_slashes() -> None:
    """Some hmmsearch dumps embed slashes in the id. The non-greedy
    regex must back off so the LAST `/start-end` is the interval."""
    out = reinterval_header(">foo/bar/100-200 desc", new_start=110, new_end=190)
    assert out == ">foo/bar/110-190 desc"


def test_reinterval_no_match_returns_unchanged() -> None:
    """Defensive fallback — a header without `/start-end` is returned
    verbatim. Parser drops malformed headers before this is called."""
    out = reinterval_header(">no_range_here desc", new_start=1, new_end=5)
    assert out == ">no_range_here desc"


# ---------------------------------------------------------------------------
# stamp_score
# ---------------------------------------------------------------------------


def test_stamp_appends_when_no_prior_score() -> None:
    out = stamp_score(">A/1-3 description", 0.1234)
    assert out == ">A/1-3 description Score=0.123"


def test_stamp_replaces_existing_score() -> None:
    out = stamp_score(">A/1-3 desc Score=9.999", 0.42)
    assert out == ">A/1-3 desc Score=0.420"


def test_stamp_is_case_insensitive_on_existing_token() -> None:
    """Lowercase `score=...` from third-party tooling also gets
    stripped — we own the canonical capitalization (`Score=`)."""
    out = stamp_score(">A/1-3 desc score=1.5 tail", 0.42)
    assert out == ">A/1-3 desc tail Score=0.420"


def test_stamp_strips_multiple_score_tokens() -> None:
    """If a caller stamped twice (rare but possible), all prior
    Score= tokens are stripped before the new one is appended."""
    out = stamp_score(">A/1-3 desc Score=0.1 mid Score=0.2 end", 0.3)
    assert out == ">A/1-3 desc mid end Score=0.300"


def test_stamp_preserves_domain_name_and_tail() -> None:
    """Worked example from PLAN §6.5: the full-domain header with all
    tail tokens survives a re-interval + score-stamp byte-for-byte
    except for the surgical edits."""
    header = ">7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1"
    out = stamp_score(reinterval_header(header, 55, 680), 0.42)
    expected = (
        ">7sch_A/55-680 [subseq from] mol:protein length:720  "
        "Exostosin-1 Score=0.420"
    )
    assert out == expected


def test_stamp_format_is_fixed_three_decimals() -> None:
    """Score format is `{:.3f}` — pinned so downstream readers have a
    stable parse."""
    assert stamp_score(">A/1-3", 0.0) == ">A/1-3 Score=0.000"
    assert stamp_score(">A/1-3", 1.5) == ">A/1-3 Score=1.500"
    assert stamp_score(">A/1-3", -0.5) == ">A/1-3 Score=-0.500"


# ---------------------------------------------------------------------------
# Composition pin — the realistic end-of-pipeline case
# ---------------------------------------------------------------------------


def test_full_realignment_render_and_header() -> None:
    """End-to-end on a synthetic alignment: OTalign produced
    columns where template residues 5..200 of the original record
    landed in query columns 30..225 (offset diagonal). After filter +
    render + header surgery, we get a 300-char row and a re-intervalled
    header.
    """
    query_len = 300
    target_seq = "X" * 250
    # Match columns: (q, t) for q in 30..225, t = q - 25 → t ∈ [5, 200].
    cols = [(q, q - 25) for q in range(30, 226)]
    # A few interior inserts that should be dropped.
    cols.insert(50, (-1, 70))
    cols.insert(60, (-1, 80))

    row = render_hit_match_only(query_len, target_seq, cols)
    assert len(row) == query_len
    assert "x" not in row  # no lowercase
    assert row[:30] == "-" * 30
    assert row[226:] == "-" * (query_len - 226)
    assert row[30:226] == "X" * 196

    span = kept_template_span(cols)
    assert span == (5, 200)

    orig_start = 100  # 1-based PDB residue start
    new_start = orig_start + span[0]  # 105
    new_end = orig_start + span[1]  # 300
    header_in = ">7sch_A/100-400 [subseq from] mol:protein length:720"
    out = stamp_score(reinterval_header(header_in, new_start, new_end), 0.875)
    assert out == (
        ">7sch_A/105-300 [subseq from] mol:protein length:720 Score=0.875"
    )
