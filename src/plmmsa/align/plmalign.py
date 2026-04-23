"""PLMAlign: affine-gap Smith-Waterman / Needleman-Wunsch on a similarity
matrix.

This module owns the DP only — scoring policy (dot / cosine / dot_zscore)
lives in `plmmsa.align.score_matrix`. The convenience `align()` entry
point inherited from `Aligner` wires a `ScoreMatrixBuilder` + this DP, so
callers still get a one-call API.

The DP itself is our own affine-gap SW/NW with explicit M/X/Y matrices.
Upstream's default pathfinder is pLM-BLAST's multi-path SW (border
traversal, no explicit gap-open/extend split); see PLAN.md for the port.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

from plmmsa.align.base import Alignment, AlignMode, MatrixAligner

# Kept at module level so existing test + downstream imports that read
# `from plmmsa.align.plmalign import SCORE_MATRIX_CHOICES` keep working.
# Canonical source is `plmmsa.align.score_matrix`.
from plmmsa.align.score_matrix import SCORE_MATRIX_CHOICES  # noqa: F401

_NEG_INF = np.float32(-1e9)

# Traceback state ids.
_M = 0  # match / mismatch
_X = 1  # gap in target (step i)
_Y = 2  # gap in query  (step j)


class PLMAlign(MatrixAligner):
    """Affine-gap SW/NW on a PLM similarity matrix."""

    id = "plmalign"
    display_name = "PLMAlign (embedding-SW/NW, affine gap)"
    default_score_matrix = "dot_zscore"

    # Kept as attributes so settings can override them at align-service
    # construction time without touching the DP code.
    DEFAULT_GAP_OPEN = 10.0
    DEFAULT_GAP_EXTEND = 1.0
    # Legacy alias — the base class now owns the value. Tests + older
    # callers that reach into the class directly still see it.
    DEFAULT_SCORE_MATRIX = "dot_zscore"

    def align_matrix(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        gap_open: float | None = None,
        gap_extend: float | None = None,
        **_: Any,
    ) -> Alignment:
        if mode not in ("local", "global"):
            raise ValueError(
                f"PLMAlign supports mode='local'|'global'; got {mode!r}. "
                "Use aligner='otalign' for glocal / q2t / t2q."
            )
        go = float(gap_open) if gap_open is not None else self.DEFAULT_GAP_OPEN
        ge = float(gap_extend) if gap_extend is not None else self.DEFAULT_GAP_EXTEND
        return _align_pair(
            np.ascontiguousarray(sim, dtype=np.float32),
            mode=mode,
            gap_open=go,
            gap_extend=ge,
        )


def _align_pair(
    sim: np.ndarray,
    *,
    mode: AlignMode,
    gap_open: float,
    gap_extend: float,
) -> Alignment:
    lq, lt = sim.shape
    go = np.float32(gap_open)
    ge = np.float32(gap_extend)

    # DP fill is O(Lq * Lt) and runs in tight numeric loops — offload to
    # numba so we get the ~50-100x speedup over pure Python. Traceback
    # stays Python: path length <= Lq + Lt makes it fast enough without
    # the JIT warmup cost.
    M, X, Y = _fill_matrices_jit(
        np.ascontiguousarray(sim, dtype=np.float32),
        go,
        ge,
        int(np.int32(1 if mode == "local" else 0)),
    )

    # Pick traceback start.
    if mode == "local":
        flat = int(np.argmax(M))
        i, j = divmod(flat, lt + 1)
        best = float(M[i, j])
        state = _M
    else:
        i, j = lq, lt
        candidates = {_M: M[i, j], _X: X[i, j], _Y: Y[i, j]}
        state = max(candidates, key=lambda k: candidates[k])
        best = float(candidates[state])

    columns = _traceback(sim, M, X, Y, i, j, state, mode=mode, go=go, ge=ge)

    if columns:
        q_idxs = [qi for qi, _ in columns if qi >= 0]
        t_idxs = [ti for _, ti in columns if ti >= 0]
        q_start, q_end = (q_idxs[0], q_idxs[-1] + 1) if q_idxs else (0, 0)
        t_start, t_end = (t_idxs[0], t_idxs[-1] + 1) if t_idxs else (0, 0)
    else:
        q_start = q_end = t_start = t_end = 0

    return Alignment(
        score=best,
        mode=mode,
        query_start=q_start,
        query_end=q_end,
        target_start=t_start,
        target_end=t_end,
        columns=columns,
    )


def _traceback(
    sim: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    i: int,
    j: int,
    state: int,
    *,
    mode: AlignMode,
    go: np.float32,
    ge: np.float32,
) -> list[tuple[int, int]]:
    columns: list[tuple[int, int]] = []
    while True:
        if mode == "local" and state == _M and M[i, j] <= 0:
            break
        if i == 0 and j == 0:
            break

        if state == _M:
            if i == 0 or j == 0:
                break
            columns.append((i - 1, j - 1))
            s = sim[i - 1, j - 1]
            prev_vals = {
                _M: float(M[i - 1, j - 1]) + float(s),
                _X: float(X[i - 1, j - 1]) + float(s),
                _Y: float(Y[i - 1, j - 1]) + float(s),
            }
            state = max(prev_vals, key=lambda k: prev_vals[k])
            i -= 1
            j -= 1
        elif state == _X:
            columns.append((i - 1, -1))
            state = _X if i > 0 and np.isclose(X[i, j], X[i - 1, j] - ge) else _M
            i -= 1
        else:  # state == _Y
            columns.append((-1, j - 1))
            state = _Y if j > 0 and np.isclose(Y[i, j], Y[i, j - 1] - ge) else _M
            j -= 1

        if i < 0 or j < 0:
            break

    columns.reverse()
    return columns


@numba.njit(
    numba.types.UniTuple(numba.float32[:, ::1], 3)(
        numba.float32[:, ::1],
        numba.float32,
        numba.float32,
        numba.int32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def _fill_matrices_jit(
    sim: np.ndarray,
    go: np.float32,
    ge: np.float32,
    is_local: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled M/X/Y affine-gap fill.

    Three matrices, each `(lq+1, lt+1)`. `is_local == 1` clamps M at 0
    (Smith-Waterman); `== 0` uses the Needleman-Wunsch boundary
    (cumulative gap penalty on the first row/col).

    `nogil=True` so the outer thread-pool fanout across targets
    actually parallelizes.
    """
    lq, lt = sim.shape
    neg_inf = np.float32(-1e9)
    M = np.full((lq + 1, lt + 1), neg_inf, dtype=np.float32)
    X = np.full((lq + 1, lt + 1), neg_inf, dtype=np.float32)
    Y = np.full((lq + 1, lt + 1), neg_inf, dtype=np.float32)

    if is_local == 1:
        for j in range(lt + 1):
            M[0, j] = np.float32(0.0)
        for i in range(lq + 1):
            M[i, 0] = np.float32(0.0)
    else:
        M[0, 0] = np.float32(0.0)
        for i in range(1, lq + 1):
            X[i, 0] = -go - (i - 1) * ge
        for j in range(1, lt + 1):
            Y[0, j] = -go - (j - 1) * ge

    for i in range(1, lq + 1):
        for j in range(1, lt + 1):
            s = sim[i - 1, j - 1]
            prev = M[i - 1, j - 1]
            v = X[i - 1, j - 1]
            if v > prev:
                prev = v
            v = Y[i - 1, j - 1]
            if v > prev:
                prev = v
            m_ij = prev + s
            if is_local == 1 and m_ij < np.float32(0.0):
                m_ij = np.float32(0.0)
            M[i, j] = m_ij

            x_open = M[i - 1, j] - go
            x_ext = X[i - 1, j] - ge
            X[i, j] = x_open if x_open > x_ext else x_ext

            y_open = M[i, j - 1] - go
            y_ext = Y[i, j - 1] - ge
            Y[i, j] = y_open if y_open > y_ext else y_ext
    return M, X, Y
