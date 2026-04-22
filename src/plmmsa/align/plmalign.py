"""Clean PLMAlign: affine-gap Smith-Waterman / Needleman-Wunsch over cosine
similarity of per-residue embeddings.

Authored fresh from the paper — not a port of the legacy plmalign_util code,
which is a pLM-BLAST-derived multi-path algorithm we can re-add later as a
second `Aligner`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from plmmsa.align.base import Aligner, Alignment, AlignMode

_NEG_INF = np.float32(-1e9)

# Traceback state ids.
_M = 0  # match / mismatch
_X = 1  # gap in target (step i)
_Y = 2  # gap in query  (step j)


class PLMAlign(Aligner):
    """Smith-Waterman / Needleman-Wunsch aligner over cosine similarity."""

    id = "plmalign"
    display_name = "PLMAlign (embedding-SW/NW, affine gap)"

    DEFAULT_GAP_OPEN = 10.0
    DEFAULT_GAP_EXTEND = 1.0
    DEFAULT_NORMALIZE = True

    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        gap_open: float | None = None,
        gap_extend: float | None = None,
        normalize: bool | None = None,
        **_: Any,
    ) -> list[Alignment]:
        go = float(gap_open) if gap_open is not None else self.DEFAULT_GAP_OPEN
        ge = float(gap_extend) if gap_extend is not None else self.DEFAULT_GAP_EXTEND
        norm = self.DEFAULT_NORMALIZE if normalize is None else bool(normalize)

        q = _normalize(query_embedding) if norm else np.asarray(query_embedding, dtype=np.float32)

        results: list[Alignment] = []
        for t_emb in target_embeddings:
            t = _normalize(t_emb) if norm else np.asarray(t_emb, dtype=np.float32)
            sim = q @ t.T
            results.append(_align_pair(sim, mode=mode, gap_open=go, gap_extend=ge))
        return results


def _normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _align_pair(
    sim: np.ndarray,
    *,
    mode: AlignMode,
    gap_open: float,
    gap_extend: float,
) -> Alignment:
    sim = np.ascontiguousarray(sim, dtype=np.float32)
    lq, lt = sim.shape
    go = np.float32(gap_open)
    ge = np.float32(gap_extend)

    # Three score matrices; +1 row/col for the empty-prefix boundary.
    M = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)
    X = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)
    Y = np.full((lq + 1, lt + 1), _NEG_INF, dtype=np.float32)

    if mode == "local":
        M[0, :] = 0.0
        M[:, 0] = 0.0
    else:
        M[0, 0] = 0.0
        for i in range(1, lq + 1):
            X[i, 0] = -go - (i - 1) * ge
        for j in range(1, lt + 1):
            Y[0, j] = -go - (j - 1) * ge

    for i in range(1, lq + 1):
        for j in range(1, lt + 1):
            s = sim[i - 1, j - 1]
            prev = max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            m_ij = prev + s
            if mode == "local" and m_ij < 0.0:
                m_ij = np.float32(0.0)
            M[i, j] = m_ij
            X[i, j] = max(M[i - 1, j] - go, X[i - 1, j] - ge)
            Y[i, j] = max(M[i, j - 1] - go, Y[i, j - 1] - ge)

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
