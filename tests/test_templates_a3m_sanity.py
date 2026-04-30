"""Parser sanity checks for the templates re-alignment pipeline.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.3. Pure-data tests, no model
load. Two flavors:

1. **Bundled fixture** — the 593-record Exostosin a3m. Pin the file-level
   invariants (query length, record count, length distribution) so a
   future fixture swap fails loudly.

2. **Hand-crafted edge cases** — small A3M strings exercising the parser
   on corner cases hmmsearch dumps actually produce: multi-line records,
   blank lines, comments, malformed headers, alphabet violations,
   interval / query-length mismatches.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pytest

from plmmsa.templates import parse_hmmsearch_a3m

_FIXTURE_PATH = Path(__file__).parent / "data" / "templates_realign" / "exostosin_hmmsearch.a3m"


# ---------------------------------------------------------------------------
# Bundled fixture
# ---------------------------------------------------------------------------


def test_fixture_passes_all_sanity_rules() -> None:
    """The bundled hmmsearch a3m parses cleanly: 593 records, 0 dropped,
    query length 649, template lengths 26..733 (median 103).

    Numbers are pinned so a fixture swap surfaces here — any future
    change to `tests/data/templates_realign/exostosin_hmmsearch.a3m`
    must be accompanied by an update to these expectations.
    """
    text = _FIXTURE_PATH.read_text()
    result = parse_hmmsearch_a3m(text)

    assert result.dropped == [], f"unexpected drops: {result.dropped[:3]}"
    assert len(result.records) == 593
    assert result.query_len == 649

    lens = [len(r.raw_seq) for r in result.records]
    assert min(lens) == 26
    assert max(lens) == 733
    assert median(lens) == 103


def test_fixture_invariants_per_record() -> None:
    """Every parsed record still satisfies the §1.2 rules at the
    record level — the parser already enforces this, so a violation
    here would be a parser bug rather than a fixture bug.
    """
    text = _FIXTURE_PATH.read_text()
    result = parse_hmmsearch_a3m(text)

    qlen = result.query_len
    for r in result.records:
        upper = sum(1 for c in r.row if c.isupper())
        lower = sum(1 for c in r.row if c.islower())
        gap = r.row.count("-")
        assert upper + gap == qlen, f"{r.target_id}: match-state count drift"
        assert upper + lower == r.end - r.start + 1, (
            f"{r.target_id}: residue count != interval"
        )
        assert len(r.raw_seq) == upper + lower
        assert r.raw_seq.isupper()


# ---------------------------------------------------------------------------
# Hand-crafted edge cases
# ---------------------------------------------------------------------------


def test_multiline_record_concatenates_correctly() -> None:
    """A record split across multiple sequence lines is reassembled.
    hmmsearch routinely wraps at 60 chars; the fixture itself wraps at
    60 too. Both records pass sanity here.
    """
    # A: 10 upper, 0 gap → query_len = 10, interval 10.
    # B: 5 upper + 2 lower + 5 gap → upper+gap = 10 (matches A),
    #    upper+lower = 7 = end-start+1.
    text = (
        ">A/1-10\n"
        "AAAAAAAAAA\n"
        ">B/1-7\n"
        "AAAA-\n"
        "Aaa--\n"
        "--\n"
    )
    result = parse_hmmsearch_a3m(text)

    assert result.dropped == []
    assert [r.target_id for r in result.records] == ["A", "B"]
    a, b = result.records
    assert a.row == "AAAAAAAAAA"
    assert b.row == "AAAA-Aaa----"
    assert b.raw_seq == "AAAAAAA"
    assert result.query_len == 10


def test_multiline_record_with_query_len_mismatch_drops() -> None:
    """Multi-line concatenation precedes the query-len check — splitting
    a row across lines must not change the verdict."""
    text = (
        ">A/1-10\n"
        "AAAAAAAAAA\n"
        ">B/1-5\n"
        "AAA\n"
        "AA\n"  # 5 upper, 0 gap → match_cols=5 ≠ 10
    )
    result = parse_hmmsearch_a3m(text)
    assert [r.target_id for r in result.records] == ["A"]
    assert len(result.dropped) == 1
    assert result.dropped[0].reason.startswith("query_len_mismatch:")


def test_blank_lines_and_comments_are_skipped() -> None:
    text = (
        "# header comment\n"
        "\n"
        ">A/10-12\n"
        "AAA\n"
        "\n"
        "# inline comment between records\n"
        ">B/5-7\n"
        "AAA\n"
    )
    result = parse_hmmsearch_a3m(text)
    assert result.dropped == []
    assert [r.target_id for r in result.records] == ["A", "B"]
    assert result.query_len == 3


def test_malformed_header_is_dropped() -> None:
    """Headers without the `/start-end` token go to `dropped`, not raised."""
    text = (
        ">A/1-3\n"
        "AAA\n"
        ">no_range_here mol:protein\n"
        "AAA\n"
    )
    result = parse_hmmsearch_a3m(text)
    assert len(result.records) == 1
    assert result.records[0].target_id == "A"
    assert len(result.dropped) == 1
    assert result.dropped[0].reason == "malformed_header"


def test_alphabet_error_is_dropped() -> None:
    text = (
        ">A/1-3\n"
        "AAA\n"
        ">B/1-3\n"
        "AB1\n"  # '1' is not a valid amino-acid letter
    )
    result = parse_hmmsearch_a3m(text)
    assert [r.target_id for r in result.records] == ["A"]
    assert [d.reason for d in result.dropped] == ["alphabet_error"]


def test_interval_mismatch_is_dropped() -> None:
    """`upper+lower != end-start+1` ⇒ interval_mismatch."""
    text = (
        ">A/1-5\n"
        "AAAAA\n"  # upper=5, lower=0, gap=0 → interval 5 ✓
        ">B/1-5\n"
        "AAA\n"  # upper=3 != 5
    )
    result = parse_hmmsearch_a3m(text)
    assert [r.target_id for r in result.records] == ["A"]
    assert len(result.dropped) == 1
    assert result.dropped[0].reason.startswith("interval_mismatch:")


def test_query_len_mismatch_is_dropped() -> None:
    """First record sets query_len; subsequent records that disagree
    on `upper+gap` are dropped."""
    text = (
        ">A/1-5\n"
        "AAAAA\n"  # upper=5, gap=0 → query_len=5
        ">B/1-3\n"
        "AAA\n"  # upper=3, gap=0 → match_cols=3, expected=5
    )
    result = parse_hmmsearch_a3m(text)
    assert [r.target_id for r in result.records] == ["A"]
    assert len(result.dropped) == 1
    assert result.dropped[0].reason.startswith("query_len_mismatch:")


def test_expected_query_len_kwarg_enforced() -> None:
    """Passing `expected_query_len` rejects records that disagree from
    the start, even the first one."""
    text = ">A/1-3\nAAA\n"
    result = parse_hmmsearch_a3m(text, expected_query_len=10)
    assert result.records == []
    assert len(result.dropped) == 1
    assert result.dropped[0].reason.startswith("query_len_mismatch:")


def test_description_preserved_for_renderer() -> None:
    """Header tail tokens (the bit downstream of `id/start-end`) survive
    parsing intact — pin the data path the renderer relies on for the
    'preserve domain name' rule."""
    header = ">7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1"
    text = f"{header}\n{'A' * 649}\n"  # 649 = query_len of the fixture
    result = parse_hmmsearch_a3m(text)
    assert len(result.records) == 1
    r = result.records[0]
    assert r.header == header
    assert r.description == "[subseq from] mol:protein length:720  Exostosin-1"
    assert r.target_id == "7sch_A"
    assert r.start == 55 and r.end == 703


def test_empty_input() -> None:
    result = parse_hmmsearch_a3m("")
    assert result.records == []
    assert result.dropped == []
    assert result.query_len == 0


def test_lowercase_residues_kept_as_inserts() -> None:
    """A row with lowercase letters parses without error — they're
    insertions (no match-state slot). Match-state count = upper + gap.
    """
    text = ">A/1-7\nAAAaaaA\n"  # upper=4, lower=3, gap=0 → query_len=4, interval=7
    result = parse_hmmsearch_a3m(text)
    assert len(result.records) == 1
    r = result.records[0]
    assert r.row == "AAAaaaA"
    assert r.raw_seq == "AAAAAAA"  # uppercased, gap-free
    assert result.query_len == 4
