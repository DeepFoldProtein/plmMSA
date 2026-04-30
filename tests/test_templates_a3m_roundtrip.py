"""Parser ↔ renderer round-trip pin for templates re-alignment.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.4. Pure-data, no model load.

For every record in the bundled fixture: derive the OTalign-style column
list from the input row (`columns_from_a3m_row`) and re-render via the
existing `plmmsa.pipeline.a3m.render_hit`. The output must equal the
input row byte-for-byte.

Why this matters: if the parser and renderer disagree on insert
placement, every OTalign-produced row that flows through the templates
pipeline inherits the same bug — and the §6.6 end-to-end test would
fail in a way that's hard to diagnose. Pinning the round-trip on
hmmsearch's own column choices isolates that risk to its actual cause.

Note: this test uses the existing `render_hit` (which keeps lowercase
inserts) — NOT the templates-specific `render_hit_match_only` (which
strips them). Insert preservation is what makes byte-equality possible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plmmsa.pipeline.a3m import AlignmentHit, render_hit
from plmmsa.templates import columns_from_a3m_row, parse_hmmsearch_a3m

_FIXTURE_PATH = Path(__file__).parent / "data" / "templates_realign" / "exostosin_hmmsearch.a3m"


def test_every_fixture_row_roundtrips_byte_equal() -> None:
    """For every record: row → columns → render_hit → row must be
    byte-identical to the original."""
    text = _FIXTURE_PATH.read_text()
    result = parse_hmmsearch_a3m(text)
    assert result.dropped == []
    qlen = result.query_len

    mismatches: list[tuple[str, str, str]] = []
    for r in result.records:
        cols = columns_from_a3m_row(r.row, qlen)
        hit = AlignmentHit(
            target_id=r.target_id,
            score=0.0,
            target_seq=r.raw_seq,
            columns=cols,
        )
        rendered = render_hit(qlen, hit)
        if rendered != r.row:
            mismatches.append((r.target_id, r.row, rendered))

    assert not mismatches, (
        f"{len(mismatches)} record(s) failed byte-equal round-trip; "
        f"first: target={mismatches[0][0]} "
        f"orig[:60]={mismatches[0][1][:60]!r} "
        f"rendered[:60]={mismatches[0][2][:60]!r}"
    )


@pytest.mark.parametrize(
    "row,query_len",
    [
        # Pure match, no inserts, no gaps.
        ("ACDEF", 5),
        # Match + interior gap.
        ("AC-EF", 5),
        # Match + insert (lowercase between match slots).
        ("ACdEF", 4),
        # Insert at the very start (before any match slot).
        ("dACEF", 4),
        # Insert at the very end (after the last match slot).
        ("ACEFd", 4),
        # Multiple inserts clustered between the same pair of slots.
        ("ACdefGH", 4),
        # Inserts surrounded by gaps.
        ("A-de-F", 4),
    ],
)
def test_handcrafted_rows_roundtrip(row: str, query_len: int) -> None:
    """Targeted edge cases for insert / gap placement — the conditions
    most likely to drift if `render_hit`'s `inserts_before` indexing or
    our `columns_from_a3m_row` increment logic gets refactored.
    """
    raw_seq = "".join(c.upper() for c in row if c != "-")
    cols = columns_from_a3m_row(row, query_len)
    hit = AlignmentHit(target_id="t", score=0.0, target_seq=raw_seq, columns=cols)
    assert render_hit(query_len, hit) == row


def test_columns_match_state_count_matches_query_len() -> None:
    """`columns_from_a3m_row` raises on row whose match-state count
    disagrees with the supplied `query_len`."""
    with pytest.raises(ValueError, match="match-state slots, expected"):
        columns_from_a3m_row("ACDEF", query_len=10)


def test_columns_alphabet_violation_raises() -> None:
    """A non-`[A-Za-z-]` character → ValueError; the parser would
    have dropped the record before we got here, but the helper still
    needs to be defensive."""
    with pytest.raises(ValueError, match="unexpected character"):
        columns_from_a3m_row("AC1EF", query_len=5)
