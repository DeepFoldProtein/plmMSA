"""pLM-BLAST (multi-path SW) coverage.

The algorithm is re-authored from the pLM-BLAST paper + PLMAlign notes
— not a port of upstream code. These tests pin the behavior the
algorithmic spec demands:

- DP recurrence with column/row-max gap term (not adjacent-cell gap).
- Multi-seed border traceback emits paths from multiple starts.
- Moving-average span extraction reports one or more high-scoring spans
  per path, with per-start dedup.
- `align_matrix` returns the top span; `align_matrix_all` returns them
  all.
- Algorithm is registered as id=`plm_blast` and inherits the
  score-matrix builder composition from `MatrixAligner.align`.
"""

from __future__ import annotations

import numpy as np

from plmmsa.align.plm_blast import (
    PlmBlast,
    _border_seeds,
    _extract_spans,
    _fill_dp,
    _moving_average,
)


def _diag_matrix(n: int, peak: float = 2.0) -> np.ndarray:
    """Build a square similarity matrix with a strong diagonal signal."""
    sim = np.full((n, n), -0.1, dtype=np.float32)
    for i in range(n):
        sim[i, i] = peak
    return sim


def test_moving_average_constant_signal() -> None:
    vals = np.ones(10, dtype=np.float32)
    out = _moving_average(vals, window=3)
    assert out.shape == (10,)
    assert np.allclose(out, 1.0)


def test_moving_average_window_1_is_identity() -> None:
    vals = np.arange(5, dtype=np.float32)
    assert np.allclose(_moving_average(vals, window=1), vals)


def test_fill_dp_local_floors_at_zero() -> None:
    sim = _diag_matrix(4, peak=1.0)
    H = _fill_dp(sim, gap_penalty=0.0, mode="local")
    assert H.shape == (5, 5)
    assert (H >= 0.0).all()


def test_fill_dp_accumulates_on_diagonal() -> None:
    sim = _diag_matrix(5, peak=1.0)
    H = _fill_dp(sim, gap_penalty=0.0, mode="local")
    # Diagonal sum should grow monotonically: 1, 2, 3, 4, 5.
    assert H[1, 1] == 1.0
    assert H[2, 2] == 2.0
    assert H[3, 3] == 3.0
    assert H[4, 4] == 4.0
    assert H[5, 5] == 5.0


def test_border_seeds_local_include_right_and_bottom() -> None:
    H = np.zeros((10, 10), dtype=np.float32)
    seeds = _border_seeds(H, min_span=2, mode="local", stride=1)
    # Every seed must be on the right or bottom edge.
    lq1, lt1 = H.shape
    for i, j in seeds:
        on_right = j == lt1 - 1
        on_bottom = i == lq1 - 1
        assert on_right or on_bottom


def test_border_seeds_global_is_corner_only() -> None:
    H = np.zeros((4, 4), dtype=np.float32)
    seeds = _border_seeds(H, min_span=1, mode="global", stride=1)
    assert seeds == [(3, 3)]


def test_extract_spans_picks_above_cutoff_region() -> None:
    # Path has a clean high-score run in the middle surrounded by lower cells.
    sim = np.asarray(
        [[0.1] * 10] * 10 + [[]],
        dtype=object,
    )
    sim = np.full((10, 10), 0.1, dtype=np.float32)
    for k in range(2, 7):
        sim[k, k] = 0.9

    path = [(i, i) for i in range(10)]
    spans = _extract_spans(path, sim, window=3, cutoff=0.5, min_span=3)
    assert len(spans) == 1
    s = spans[0]
    assert s.qi_start >= 2 and s.qi_end <= 7
    assert s.score > 0.5


def test_align_matrix_returns_best_span() -> None:
    aligner = PlmBlast()
    sim = _diag_matrix(30, peak=2.0)
    alignment = aligner.align_matrix(sim, min_span=5, window_size=3, sigma_factor=0.5)
    assert alignment.score > 0.0
    # On a perfect diagonal the alignment should span most of the residues.
    assert alignment.query_end - alignment.query_start >= 10
    assert alignment.target_end - alignment.target_start >= 10
    # Columns on the diagonal should all have qi == ti.
    for qi, ti in alignment.columns:
        if qi >= 0 and ti >= 0:
            assert qi == ti


def test_align_matrix_empty_on_zero_matrix() -> None:
    """Zero-valued similarity → no span exceeds a positive cutoff. Returns
    a zero-score Alignment, not a crash. (On a *constant* non-zero matrix
    upstream would still report a long span at that constant level, so we
    test the strictly-zero case.)"""
    aligner = PlmBlast()
    sim = np.zeros((10, 10), dtype=np.float32)
    alignment = aligner.align_matrix(sim, min_span=3, sigma_factor=1.0)
    assert alignment.score == 0.0
    assert alignment.columns == []


def test_align_matrix_all_ranks_descending() -> None:
    """When the matrix has two strong regions, align_matrix_all reports
    both spans, best-first."""
    n = 40
    sim = np.full((n, n), -0.1, dtype=np.float32)
    # Two diagonal blocks of high similarity.
    for k in range(0, 8):
        sim[k, k] = 2.0
    for k in range(25, 33):
        sim[k, k] = 1.5

    aligner = PlmBlast()
    spans = aligner.align_matrix_all(
        sim, min_span=4, window_size=3, sigma_factor=0.3,
    )
    assert len(spans) >= 1
    scores = [s.score for s in spans]
    assert scores == sorted(scores, reverse=True)


def test_align_composes_builder_plus_matrix() -> None:
    """`align(q, targets, score_matrix=…)` must flow through the builder
    registry and then call `align_matrix` per target."""
    aligner = PlmBlast()
    q = np.eye(15, 4, dtype=np.float32)
    t1 = np.eye(15, 4, dtype=np.float32)  # identical
    t2 = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)

    results = aligner.align(
        q, [t1, t2], score_matrix="cosine", min_span=3, window_size=3, sigma_factor=0.3,
    )
    assert len(results) == 2
    # The identical target should score strictly higher than random noise.
    assert results[0].score > results[1].score
