"""PLMAlign score-matrix mode coverage.

Three modes — dot_zscore (default, upstream-compatible), cosine (v0
default), and dot (raw) — produce different absolute scores for the same
embeddings. These tests exercise each path and confirm:

- Option selection works from both the explicit `score_matrix` kwarg and
  the legacy `normalize=True/False` alias.
- Invalid mode names raise ValueError with the allowed set.
- Z-score mode handles the degenerate all-equal-similarity case (the
  `+1e-3` epsilon on std keeps it finite instead of producing NaNs).
"""

from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align.plmalign import SCORE_MATRIX_CHOICES, PLMAlign


def _query_target() -> tuple[np.ndarray, np.ndarray]:
    # A 3x4 query and a 4x4 target with the first three rows matching.
    q = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    t = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return q, t


def test_default_is_dot_zscore() -> None:
    assert PLMAlign.DEFAULT_SCORE_MATRIX == "dot_zscore"


def test_all_three_modes_run() -> None:
    q, t = _query_target()
    aligner = PLMAlign()
    for mode in SCORE_MATRIX_CHOICES:
        alignments = aligner.align(q, [t], score_matrix=mode)
        assert len(alignments) == 1
        # Every mode should recognize the diagonal match.
        al = alignments[0]
        matched = [(qi, ti) for qi, ti in al.columns if qi >= 0 and ti >= 0]
        assert (0, 0) in matched
        assert (1, 1) in matched
        assert (2, 2) in matched


def test_cosine_score_bounded() -> None:
    q, t = _query_target()
    aligner = PLMAlign()
    [al] = aligner.align(q, [t], score_matrix="cosine", mode="local")
    # Score is now `mean(raw_similarity[path])` (upstream PLMAlign
    # convention, see align.plmalign._align_pair). Three perfect cosine
    # matches → mean ≈ 1.0.
    assert 0.9 <= al.score <= 1.1


def test_dot_and_zscore_share_raw_scoring() -> None:
    """Upstream-compatible scoring: both `dot` and `dot_zscore` score
    the path against the raw dot-product matrix
    (`raw_similarity_for_scoring`), so on inputs where the DP picks the
    same path under raw and Z-scored DP fills (the typical case for a
    clean diagonal), the reported scores match.
    """
    q, t = _query_target()
    aligner = PLMAlign()
    [raw] = aligner.align(q, [t], score_matrix="dot")
    [zscored] = aligner.align(q, [t], score_matrix="dot_zscore")
    assert np.isclose(raw.score, zscored.score)


def test_legacy_normalize_kwarg_maps_to_cosine() -> None:
    q, t = _query_target()
    aligner = PLMAlign()
    [via_cosine] = aligner.align(q, [t], score_matrix="cosine")
    [via_legacy] = aligner.align(q, [t], normalize=True)
    assert np.isclose(via_legacy.score, via_cosine.score)


def test_legacy_normalize_false_maps_to_dot() -> None:
    q, t = _query_target()
    aligner = PLMAlign()
    [via_dot] = aligner.align(q, [t], score_matrix="dot")
    [via_legacy] = aligner.align(q, [t], normalize=False)
    assert np.isclose(via_legacy.score, via_dot.score)


def test_invalid_mode_raises() -> None:
    q, t = _query_target()
    aligner = PLMAlign()
    with pytest.raises(ValueError, match="score_matrix must be one of"):
        aligner.align(q, [t], score_matrix="softmax_attention")


def test_zscore_degenerate_matrix_does_not_nan() -> None:
    # All-equal similarity matrix → std is zero. The +1e-3 floor keeps the
    # Z-score finite; the aligner should still produce a result.
    q = np.ones((3, 4), dtype=np.float32)
    t = np.ones((4, 4), dtype=np.float32)
    aligner = PLMAlign()
    [al] = aligner.align(q, [t], score_matrix="dot_zscore")
    assert np.isfinite(al.score)
