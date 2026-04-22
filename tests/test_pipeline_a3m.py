from __future__ import annotations

from plmmsa.pipeline.a3m import AlignmentHit, assemble_a3m, render_hit


def test_render_hit_identity_row() -> None:
    query = "MKTA"
    target = "MKTA"
    hit = AlignmentHit(
        target_id="T",
        score=4.0,
        target_seq=target,
        columns=[(0, 0), (1, 1), (2, 2), (3, 3)],
    )
    assert render_hit(len(query), hit) == "MKTA"


def test_render_hit_gap_in_target() -> None:
    query = "MKTA"
    target = "MKA"  # target missing T at position 2
    hit = AlignmentHit(
        target_id="T",
        score=3.0,
        target_seq=target,
        columns=[(0, 0), (1, 1), (2, -1), (3, 2)],
    )
    assert render_hit(len(query), hit) == "MK-A"


def test_render_hit_insertion_is_lowercase() -> None:
    query = "MKA"
    target = "MKxA"  # target has an extra residue between K and A
    hit = AlignmentHit(
        target_id="T",
        score=3.0,
        target_seq=target,
        columns=[(0, 0), (1, 1), (-1, 2), (2, 3)],
    )
    assert render_hit(len(query), hit) == "MKxA"


def test_render_hit_partial_alignment_pads_with_gaps() -> None:
    query = "MKTIIAL"
    target = "TI"
    # Local alignment covering only query cols 2..3.
    hit = AlignmentHit(
        target_id="T",
        score=2.0,
        target_seq=target,
        columns=[(2, 0), (3, 1)],
    )
    assert render_hit(len(query), hit) == "--TI---"


def test_assemble_a3m_writes_query_first() -> None:
    query = "MKT"
    hits = [
        AlignmentHit(
            target_id="H1",
            score=2.5,
            target_seq="MKT",
            columns=[(0, 0), (1, 1), (2, 2)],
        ),
    ]
    a3m = assemble_a3m(query_id="Q", query_seq=query, query_self_score=3.0, hits=hits)
    lines = a3m.rstrip("\n").splitlines()
    assert lines[0] == ">Q   3.000"
    assert lines[1] == "MKT"
    assert lines[2] == ">H1   2.500"
    assert lines[3] == "MKT"


def test_assemble_a3m_empty_hits_emits_query_only() -> None:
    a3m = assemble_a3m(query_id="Q", query_seq="MKT", query_self_score=3.0, hits=[])
    assert a3m == ">Q   3.000\nMKT\n"
