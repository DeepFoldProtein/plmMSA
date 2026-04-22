"""Round-trip our A3M assembler against the upstream format.

We parse the T1120 example (`tests/fixtures/example_a3m.txt`, lifted from
upstream `DeepFold-PLM/plmMSA/example_fastas/msas/...`), reconstruct the
`AlignmentHit` records that would have produced each rendered row, and feed
them back through `assemble_a3m`. If the result diverges from the fixture
byte-for-byte we've either broken the convention or the fixture is stale —
either way we want CI to flag it.
"""

from __future__ import annotations

from pathlib import Path

from plmmsa.pipeline.a3m import AlignmentHit, assemble_a3m, render_hit

_FIXTURE = Path(__file__).parent / "fixtures" / "example_a3m.txt"


def _parse_row(row: str) -> tuple[str, list[tuple[int, int]]]:
    """Parse a rendered A3M row.

    Returns `(target_seq, columns)` where `target_seq` is the concatenation
    of every residue that appears in the row (uppercase match letters +
    lowercase insertions, in order), and `columns` is the full alignment
    path as `(query_idx, target_idx)` pairs (`-1` marks a gap on that side).
    """
    target_chars: list[str] = []
    columns: list[tuple[int, int]] = []
    qi = 0
    ti = 0
    for char in row:
        if char == "-":
            columns.append((qi, -1))
            qi += 1
        elif char.isupper():
            columns.append((qi, ti))
            target_chars.append(char)
            qi += 1
            ti += 1
        elif char.islower():
            columns.append((-1, ti))
            target_chars.append(char.upper())
            ti += 1
        else:
            raise ValueError(f"unexpected A3M character {char!r}")
    return "".join(target_chars), columns


def _parse_header(line: str) -> tuple[str, float]:
    assert line.startswith(">"), line
    name, _, score = line[1:].strip().rpartition(" ")
    name = name.strip()
    return name, float(score.strip())


def test_render_hit_matches_fixture_byte_for_byte() -> None:
    raw = _FIXTURE.read_text()
    lines = raw.splitlines()
    query_id, query_score = _parse_header(lines[0])
    query_seq = lines[1]

    hits: list[AlignmentHit] = []
    for i in range(2, len(lines), 2):
        target_id, score = _parse_header(lines[i])
        target_seq, columns = _parse_row(lines[i + 1])
        hits.append(
            AlignmentHit(
                target_id=target_id,
                score=score,
                target_seq=target_seq,
                columns=columns,
            )
        )
        # Per-row round-trip: our renderer has to reproduce exactly the row
        # we parsed from the fixture.
        assert render_hit(len(query_seq), hits[-1]) == lines[i + 1], target_id

    rebuilt = assemble_a3m(
        query_id=query_id,
        query_seq=query_seq,
        query_self_score=query_score,
        hits=hits,
    )
    # Preserve the trailing newline convention from the fixture.
    assert rebuilt == raw


def test_parser_extracts_expected_hit_count() -> None:
    raw = _FIXTURE.read_text().splitlines()
    headers = [ln for ln in raw if ln.startswith(">")]
    # Query + two hits per the upstream example.
    assert len(headers) == 3
    assert headers[0].startswith(">T1120")
    assert headers[1].startswith(">Q9RYE6")
    assert headers[2].startswith(">A0A841HZG6")
