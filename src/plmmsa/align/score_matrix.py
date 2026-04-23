"""Score-matrix construction, decoupled from the aligner.

Separation of concerns:

- A `ScoreMatrixBuilder` takes raw per-residue embeddings and produces one
  similarity matrix per target. That's it — no alignment, no traceback.
- An `Aligner` consumes a similarity matrix (see `Aligner.align_matrix`)
  and produces an `Alignment`. It never touches embeddings directly.

This split has two payoffs:

1. Every aligner gets every scoring mode for free. When pLM-BLAST's
   multi-path SW lands it reuses `NumpyDotZScoreBuilder` verbatim;
   OTalign likewise.
2. Matrix construction can move to GPU independently of the DP. A future
   `TorchGPUBuilder` keeps tensors on-device and returns cpu numpy right
   before the serial DP, which is the stage that can't be parallelized.
   The aligner changes not a line.

Wire format for picking a builder is `score_matrix` in the aligner
options (also settable via `settings.aligners.plmalign.score_matrix`). See
`SCORE_MATRIX_CHOICES` for the accepted keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

SCORE_MATRIX_CHOICES = ("dot_zscore", "cosine", "dot")


@runtime_checkable
class ScoreMatrixBuilder(Protocol):
    """Contract for anything that turns raw embeddings into similarity matrices."""

    id: str

    def build(
        self, query: np.ndarray, targets: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """Return one `[Lq, Lt_i]` similarity matrix per target.

        Must preserve target order and must return np.ndarray (float32) so
        the aligner's DP can consume it directly. Builders that use GPU
        internally are expected to move results back to CPU here.
        """
        ...


class DotBuilder:
    """Raw dot product, no normalization. Primarily a debug tool.

    Score scale tracks embedding magnitudes and will differ wildly across
    PLMs — not meaningful to compare jobs that use different models.
    """

    id = "dot"

    def build(
        self, query: np.ndarray, targets: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        return [q @ np.asarray(t, dtype=np.float32).T for t in targets]


class CosineBuilder:
    """L2-normalize each per-residue vector, then dot. Scores in [-1, 1].

    Robust to magnitude drift across PLMs; was our v0 default. Simpler to
    reason about when tuning gap penalties than the Z-scored variant.
    """

    id = "cosine"

    def build(
        self, query: np.ndarray, targets: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        q = _l2_normalize(query)
        return [q @ _l2_normalize(t).T for t in targets]


class DotZScoreBuilder:
    """Raw dot product, then global Z-score per target matrix.

    Matches upstream PLMAlign's scoring convention. Centers each pairwise
    similarity matrix on zero with ~unit variance; the `+1e-3` floor on
    std keeps the result finite when the matrix is degenerate.
    """

    id = "dot_zscore"

    def build(
        self, query: np.ndarray, targets: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        out: list[np.ndarray] = []
        for t in targets:
            sim = q @ np.asarray(t, dtype=np.float32).T
            out.append(_zscore(sim))
        return out


_REGISTRY: dict[str, ScoreMatrixBuilder] = {
    DotZScoreBuilder.id: DotZScoreBuilder(),
    CosineBuilder.id: CosineBuilder(),
    DotBuilder.id: DotBuilder(),
}


def get_builder(score_matrix: str) -> ScoreMatrixBuilder:
    """Look up a builder by its `score_matrix` id.

    Raises `ValueError` with the accepted set so callers can surface the
    failure as a clear 400 at the service boundary.
    """
    if score_matrix not in _REGISTRY:
        raise ValueError(
            f"score_matrix must be one of {SCORE_MATRIX_CHOICES}, got {score_matrix!r}"
        )
    return _REGISTRY[score_matrix]


def register_builder(builder: ScoreMatrixBuilder) -> None:
    """Register an additional builder at runtime.

    Exists so a GPU-backed builder can slot itself in without editing this
    module. Keyed on `builder.id`; collisions overwrite silently — the
    caller is expected to pick a distinct id.
    """
    _REGISTRY[builder.id] = builder


# --- internals ---------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _zscore(s: np.ndarray) -> np.ndarray:
    mean = float(s.mean())
    std = float(s.std())
    return ((s - mean) / (std + 1e-3)).astype(np.float32)


__all__ = [
    "SCORE_MATRIX_CHOICES",
    "CosineBuilder",
    "DotBuilder",
    "DotZScoreBuilder",
    "ScoreMatrixBuilder",
    "get_builder",
    "register_builder",
]
