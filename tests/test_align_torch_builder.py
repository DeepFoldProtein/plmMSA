"""Torch-backed ScoreMatrixBuilder coverage (runs on CPU torch).

Exercises the same three scoring modes against the numpy builders to
confirm they agree to within f32 rounding. Uses `device="cpu"` so this
test runs anywhere torch is installed; the GPU path is the same math
moved to CUDA tensors.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from plmmsa.align import score_matrix as numpy_sm
from plmmsa.align.torch_score_matrix import TorchBuilder, register_all


def _qt() -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(12, 8)).astype(np.float32)
    targets = [rng.normal(size=(n, 8)).astype(np.float32) for n in (10, 14)]
    return q, targets


@pytest.mark.parametrize("mode", ["dot", "cosine", "dot_zscore"])
def test_torch_builder_matches_numpy(mode: str) -> None:
    q, targets = _qt()
    torch_out = TorchBuilder(mode, device="cpu").build(q, targets)
    numpy_out = numpy_sm.get_builder(mode).build(q, targets)
    assert len(torch_out) == len(numpy_out)
    for t_mat, n_mat in zip(torch_out, numpy_out, strict=True):
        assert t_mat.shape == n_mat.shape
        assert np.allclose(t_mat, n_mat, atol=1e-5)


def test_torch_builder_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        TorchBuilder("softmax", device="cpu")


def test_register_all_swaps_registry_entries() -> None:
    """After `register_all('cpu')`, `get_builder('dot_zscore')` returns the
    torch builder. Restored afterwards to keep other tests stable."""
    # Snapshot before, swap, assert swap, restore.
    before = {m: numpy_sm.get_builder(m) for m in ("dot", "cosine", "dot_zscore")}
    try:
        register_all(device="cpu")
        after = {m: numpy_sm.get_builder(m) for m in ("dot", "cosine", "dot_zscore")}
        for m in after:
            assert isinstance(after[m], TorchBuilder)
    finally:
        for b in before.values():
            numpy_sm.register_builder(b)
