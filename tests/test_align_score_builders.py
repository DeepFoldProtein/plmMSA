"""Coverage for the `ScoreMatrixBuilder` registry.

The `align.score_matrix` module is the new home for scoring policy —
keeping it tested independently of the aligner makes it easy to add a
GPU-backed builder later without touching the DP.
"""

from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align import score_matrix as sm
from plmmsa.align.base import Alignment, AlignMode, MatrixAligner


def _qt() -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    return q, t


def test_registry_has_three_choices() -> None:
    assert set(sm.SCORE_MATRIX_CHOICES) == {"dot_zscore", "cosine", "dot"}
    for choice in sm.SCORE_MATRIX_CHOICES:
        assert sm.get_builder(choice).id == choice


def test_unknown_choice_raises() -> None:
    with pytest.raises(ValueError, match="score_matrix must be one of"):
        sm.get_builder("softmax_attention")


def test_dot_builder_is_raw_product() -> None:
    q, t = _qt()
    [out] = sm.DotBuilder().build(q, [t])
    assert out.shape == (2, 3)
    assert np.allclose(out, q @ t.T)


def test_cosine_builder_scores_bounded() -> None:
    q, t = _qt()
    [out] = sm.CosineBuilder().build(q, [t])
    assert np.all(out <= 1.0 + 1e-6)
    assert np.all(out >= -1.0 - 1e-6)
    # Unit-length inputs → cosine(0,0) == 1.
    assert np.isclose(out[0, 0], 1.0)


def test_dot_zscore_centers_and_rescales() -> None:
    q, t = _qt()
    [out] = sm.DotZScoreBuilder().build(q, [t])
    # Std + 1e-3 epsilon means we don't get exactly zero mean, but close.
    assert abs(float(out.mean())) < 1e-5
    # Reject nan/inf on degenerate inputs too.
    flat = np.ones((2, 2), dtype=np.float32)
    [degenerate] = sm.DotZScoreBuilder().build(flat, [flat])
    assert np.all(np.isfinite(degenerate))


def test_register_additional_builder() -> None:
    """Third-party builder can slot into the registry at runtime."""

    class _Doubler:
        id = "doubler"

        def build(self, query, targets):
            return [2.0 * (query @ np.asarray(t).T) for t in targets]

    sm.register_builder(_Doubler())
    try:
        out = sm.get_builder("doubler").build(*_qt_list())
        assert out[0].shape == (2, 3)
        assert np.allclose(out[0], 2.0 * _qt()[0] @ _qt()[1].T)
    finally:
        # Best-effort cleanup so other tests see a predictable registry.
        sm._REGISTRY.pop("doubler", None)  # pyright: ignore[reportPrivateUsage]


def _qt_list() -> tuple[np.ndarray, list[np.ndarray]]:
    q, t = _qt()
    return q, [t]


class _CollectorAligner(MatrixAligner):
    """Aligner that records every similarity matrix it sees — proves the
    base-class default really does call `align_matrix` per target."""

    id = "collector"
    display_name = "test-collector"

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def align_matrix(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        **_: object,
    ) -> Alignment:
        self.seen.append(sim.copy())
        return Alignment(
            score=float(sim.max()),
            mode=mode,
            query_start=0,
            query_end=sim.shape[0],
            target_start=0,
            target_end=sim.shape[1],
        )


def test_base_align_routes_through_builder_then_align_matrix() -> None:
    q, t = _qt()
    a = _CollectorAligner()
    results = a.align(q, [t, t], score_matrix="cosine")
    assert len(results) == 2
    assert len(a.seen) == 2
    # CosineBuilder normalizes → max cell value is 1.0 on unit-length input.
    assert np.isclose(a.seen[0].max(), 1.0)


def test_base_align_legacy_normalize_kwarg() -> None:
    q, t = _qt()
    a = _CollectorAligner()
    a.align(q, [t], normalize=True)
    # With cosine, peak is bounded by 1.
    assert a.seen[-1].max() <= 1.0 + 1e-6

    a.align(q, [t], normalize=False)
    # With raw dot, peak equals q @ t.T max (can exceed 1 when inputs aren't normed).
    assert a.seen[-1].max() >= 0.0
