"""OTalign affine-gap DP with per-residue gap costs.

Consumes:
  - `S` : `(Lq, Lt)` per-cell match score (PMI from the transport plan).
  - `go_q, ge_q` : `(Lq,)` position-specific gap-open / gap-extend on the
                   query side (applied when stepping to `(i-1, j)` — i.e.,
                   a gap in the target).
  - `go_t, ge_t` : `(Lt,)` same on the target side.
  - `mode`       : alignment mode, one of the five below.

Produces one traceback: a list of `(qi, ti)` pairs (gaps as `-1`) and the
final path score. The aligner's glue module converts that into an
`Alignment` and computes the headline OT-plan-sum score.

Three-state affine gap (M/X/Y) recurrence, no scoring math lives here —
`S` is pre-computed. Five modes change only initialization (of the
boundary rows) and the traceback start / end conditions:

  global    — NW. `M[0, 0] = 0`; boundary rows pay cumulative gap cost.
              Traceback starts at `(Lq, Lt)`.
  q2t       — query-open. `M[i, 0] = 0` free; `M[0, j]` penalized.
              Traceback starts at `argmax_j M[Lq, j]` (last row).
  t2q       — template-open. Mirror of q2t.
  glocal    — semi-global. Free end-gaps on both sides. `M[i, 0] = 0`
              and `M[0, j] = 0`; traceback starts at `argmax M[:, :]`
              over last row + last column.
  local     — SW. Floors non-M values at 0; traceback starts at the
              full-matrix argmax and stops when the cell hits 0.

The DP is O(Lq * Lt) — standard. Per-residue gap costs add no asymptotic
overhead, just one indexed lookup per cell update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np

DPMode = Literal["global", "q2t", "t2q", "glocal", "local"]

_NEG_INF = np.float32(-1e9)

# Traceback state tags. M = match/mismatch, X = gap in target, Y = gap in query.
_M = 0
_X = 1
_Y = 2


@numba.njit(
    numba.void(
        numba.float32[:, ::1],  # S
        numba.float32[::1],  # go_q
        numba.float32[::1],  # ge_q
        numba.float32[::1],  # go_t
        numba.float32[::1],  # ge_t
        numba.float32[:, ::1],  # M (in/out)
        numba.float32[:, ::1],  # X (in/out)
        numba.float32[:, ::1],  # Y (in/out)
        numba.boolean,  # is_local
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def _fill_matrices_jit(S, go_q, ge_q, go_t, ge_t, M, X, Y, is_local):
    """Main DP fill — 3-state affine-gap recurrence with per-residue gap costs.

    Numba-compiled; ~30-50x faster than the plain-numpy version. Same
    math as the docstring recurrence; the `mode == "local"` check from
    the Python version is lifted to a boolean parameter so the inner
    loop has no string compares.
    """
    lq = S.shape[0]
    lt = S.shape[1]
    for i in range(1, lq + 1):
        open_cost_q = go_q[i - 1]
        ext_cost_q = ge_q[i - 1]
        for j in range(1, lt + 1):
            s_ij = S[i - 1, j - 1]

            m_prev = M[i - 1, j - 1]
            x_prev = X[i - 1, j - 1]
            y_prev = Y[i - 1, j - 1]
            prev = m_prev if m_prev > x_prev else x_prev
            if y_prev > prev:
                prev = y_prev
            m_val = prev + s_ij
            if is_local and m_val < 0.0:
                m_val = np.float32(0.0)
            M[i, j] = m_val

            # Gap in target ⇒ deletion on query side at residue i-1.
            x_from_m = M[i - 1, j] - open_cost_q
            x_from_x = X[i - 1, j] - ext_cost_q
            X[i, j] = x_from_m if x_from_m > x_from_x else x_from_x

            # Gap in query ⇒ insertion on target side at residue j-1.
            open_cost_t = go_t[j - 1]
            ext_cost_t = ge_t[j - 1]
            y_from_m = M[i, j - 1] - open_cost_t
            y_from_y = Y[i, j - 1] - ext_cost_t
            Y[i, j] = y_from_m if y_from_m > y_from_y else y_from_y


@dataclass(slots=True)
class DPResult:
    """Alignment produced by the DP, independent of the scoring module.

    The glue module converts this into an `Alignment` after computing
    the OT-plan-sum score — so this struct doesn't carry the headline
    score, only the path and the DP's own objective (useful for
    debugging + comparing runs across modes)."""

    columns: list[tuple[int, int]]  # (qi, ti) pairs, gaps as -1
    path_score: float  # sum of M-state visits (PMI)
    q_start: int
    q_end: int  # exclusive
    t_start: int
    t_end: int


def affine_gap_dp(
    S: np.ndarray,
    *,
    go_q: np.ndarray,
    ge_q: np.ndarray,
    go_t: np.ndarray,
    ge_t: np.ndarray,
    mode: DPMode = "glocal",
) -> DPResult:
    """Run the per-cell affine-gap DP and return the traceback.

    Shapes:
      S : (Lq, Lt) float
      go_q / ge_q : (Lq,) float
      go_t / ge_t : (Lt,) float
    """
    S = np.ascontiguousarray(S, dtype=np.float32)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D, got shape {S.shape}")
    lq, lt = S.shape
    if lq == 0 or lt == 0:
        return DPResult(columns=[], path_score=0.0, q_start=0, q_end=0, t_start=0, t_end=0)

    go_q = np.ascontiguousarray(go_q, dtype=np.float32)
    ge_q = np.ascontiguousarray(ge_q, dtype=np.float32)
    go_t = np.ascontiguousarray(go_t, dtype=np.float32)
    ge_t = np.ascontiguousarray(ge_t, dtype=np.float32)
    if go_q.shape != (lq,) or ge_q.shape != (lq,):
        raise ValueError(f"go_q / ge_q must have shape ({lq},)")
    if go_t.shape != (lt,) or ge_t.shape != (lt,):
        raise ValueError(f"go_t / ge_t must have shape ({lt},)")

    M = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)
    X = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)
    Y = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)

    _initialize_boundaries(M, X, Y, mode, go_q, ge_q, go_t, ge_t)

    _fill_matrices_jit(S, go_q, ge_q, go_t, ge_t, M, X, Y, mode == "local")

    start_i, start_j, start_state, best_score = _select_traceback_start(
        M,
        X,
        Y,
        mode,
    )
    columns = _traceback(
        S,
        M,
        X,
        Y,
        start_i,
        start_j,
        start_state,
        mode=mode,
        go_q=go_q,
        ge_q=ge_q,
        go_t=go_t,
        ge_t=ge_t,
    )

    if columns:
        q_idxs = [qi for qi, _ in columns if qi >= 0]
        t_idxs = [ti for _, ti in columns if ti >= 0]
        q_start = q_idxs[0] if q_idxs else 0
        q_end = (q_idxs[-1] + 1) if q_idxs else 0
        t_start = t_idxs[0] if t_idxs else 0
        t_end = (t_idxs[-1] + 1) if t_idxs else 0
    else:
        q_start = q_end = t_start = t_end = 0

    return DPResult(
        columns=columns,
        path_score=float(best_score),
        q_start=q_start,
        q_end=q_end,
        t_start=t_start,
        t_end=t_end,
    )


def _initialize_boundaries(
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mode: DPMode,
    go_q: np.ndarray,
    ge_q: np.ndarray,
    go_t: np.ndarray,
    ge_t: np.ndarray,
) -> None:
    """Seed the boundary rows / columns based on the mode.

    Local mode: everything zero — no penalty for starting anywhere.
    Global/q2t/t2q/glocal: match the operator's description of free vs.
    paid end-gaps on each side.
    """
    lq = M.shape[0] - 1
    lt = M.shape[1] - 1

    if mode == "local":
        # Every cell can start an alignment at zero.
        M[0, :] = 0.0
        M[:, 0] = 0.0
        return

    # In non-local modes we always start somewhere with score 0; what
    # differs is whether boundary moves pay gap cost.
    M[0, 0] = 0.0

    # Top row (Y): inserting target residues before the query begins.
    # `global` and `q2t` penalize these; `t2q` and `glocal` don't.
    pay_top = mode in ("global", "q2t")
    if pay_top:
        cum = 0.0
        for j in range(1, lt + 1):
            # First column gap pays open; later cells pay extend.
            cum += float(go_t[j - 1]) if j == 1 else float(ge_t[j - 1])
            Y[0, j] = -cum
    else:
        for j in range(1, lt + 1):
            Y[0, j] = 0.0
            M[0, j] = 0.0

    # Left column (X): deleting query residues before the target begins.
    # `global` and `t2q` penalize; `q2t` and `glocal` don't.
    pay_left = mode in ("global", "t2q")
    if pay_left:
        cum = 0.0
        for i in range(1, lq + 1):
            cum += float(go_q[i - 1]) if i == 1 else float(ge_q[i - 1])
            X[i, 0] = -cum
    else:
        for i in range(1, lq + 1):
            X[i, 0] = 0.0
            M[i, 0] = 0.0


def _select_traceback_start(
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mode: DPMode,
) -> tuple[int, int, int, float]:
    lq = M.shape[0] - 1
    lt = M.shape[1] - 1

    if mode == "local":
        # Argmax over the full M table.
        flat = int(np.argmax(M))
        i, j = divmod(flat, M.shape[1])
        return i, j, _M, float(M[i, j])

    if mode == "global":
        # Bottom-right corner, pick the state with the highest score.
        i, j = lq, lt
        candidates = {_M: float(M[i, j]), _X: float(X[i, j]), _Y: float(Y[i, j])}
        state = max(candidates, key=lambda k: candidates[k])
        return i, j, state, candidates[state]

    if mode == "q2t":
        # Last row (i = lq), any j — free right-end gap on query side.
        j = int(np.argmax(M[lq, :]))
        return lq, j, _M, float(M[lq, j])

    if mode == "t2q":
        # Last column (j = lt), any i.
        i = int(np.argmax(M[:, lt]))
        return i, lt, _M, float(M[i, lt])

    # glocal: best across the last row and last column.
    j_best_row = int(np.argmax(M[lq, :]))
    i_best_col = int(np.argmax(M[:, lt]))
    m_row = float(M[lq, j_best_row])
    m_col = float(M[i_best_col, lt])
    if m_row >= m_col:
        return lq, j_best_row, _M, m_row
    return i_best_col, lt, _M, m_col


def _traceback(
    S: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    i: int,
    j: int,
    state: int,
    *,
    mode: DPMode,
    go_q: np.ndarray,
    ge_q: np.ndarray,
    go_t: np.ndarray,
    ge_t: np.ndarray,
) -> list[tuple[int, int]]:
    """Walk back to the origin (or a zero cell in local mode)."""
    columns: list[tuple[int, int]] = []
    while True:
        # Local mode stops when we hit a zero M cell.
        if mode == "local" and state == _M and M[i, j] <= 0.0:
            break
        if i == 0 and j == 0:
            break

        if state == _M:
            if i == 0 or j == 0:
                break
            columns.append((i - 1, j - 1))
            s_ij = float(S[i - 1, j - 1])
            # Which predecessor produced M[i,j]? Pick the argmax.
            prev_vals = {
                _M: float(M[i - 1, j - 1]) + s_ij,
                _X: float(X[i - 1, j - 1]) + s_ij,
                _Y: float(Y[i - 1, j - 1]) + s_ij,
            }
            state = max(prev_vals, key=lambda k: prev_vals[k])
            i -= 1
            j -= 1
        elif state == _X:
            columns.append((i - 1, -1))
            # Did X[i,j] come from an open (from M) or an extend (from X)?
            open_ = float(M[i - 1, j]) - float(go_q[i - 1])
            ext = float(X[i - 1, j]) - float(ge_q[i - 1])
            state = _X if ext > open_ else _M
            i -= 1
        else:  # state == _Y
            columns.append((-1, j - 1))
            open_ = float(M[i, j - 1]) - float(go_t[j - 1])
            ext = float(Y[i, j - 1]) - float(ge_t[j - 1])
            state = _Y if ext > open_ else _M
            j -= 1

        if i < 0 or j < 0:
            break

    columns.reverse()
    return columns


__all__ = ["DPMode", "DPResult", "affine_gap_dp"]
