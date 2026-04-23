"""pLM-BLAST: multi-path Smith-Waterman over a PLM similarity matrix.

Re-authored from the pLM-BLAST paper + PLMAlign reference notes — NOT a
direct port of the upstream code. The algorithmic spec this implementation
follows, in plain terms:

1. **DP recurrence (local mode).** Same cell update as plain local SW *with
   one twist*: the gap term takes the max over the entire preceding column
   (for query-side gaps) or row (for target-side gaps), not just the
   adjacent cell. This gives cheaper "leap" gaps than the classic
   `H[i-1,j] - gap_extend` term. Specifically:

       H[i,j] = max(
           0,                                      # local floor
           H[i-1, j-1] + sim[i-1, j-1],            # diagonal
           max(H[1:i, j]) - gap_penalty,           # gap in query
           max(H[i, 1:j]) - gap_penalty,           # gap in target
       )

   Gap is a single linear penalty — no affine open/extend split. Matches
   upstream's `GAP_OPEN=0, GAP_EXT=0` default; operators override via
   request options.

2. **Border seeding.** After the DP fills, seed traceback from cells on the
   bottom and right edges of H (excluding a `min_span` corner that can't
   host a reportable alignment). Each seed runs its own traceback → one
   candidate path.

3. **Path scoring via moving average.** After traceback, walk the path and
   compute a moving mean over a window of `window_size` cells. Contiguous
   runs where the moving mean is above `sigma_factor * sigma(sim)` become
   one reported "span". One path can yield zero, one, or several spans.

4. **Output selection.** All spans across all paths go through a dedupe
   step: group by start coordinate `(qi, ti)`, keep the highest-mean
   span per group, sort descending. The aligner's `align_matrix` contract
   returns the single best span; the full list is available via
   `align_matrix_all`.

Algorithm is O(Lq * Lt * max(Lq, Lt)) because the column/row-max gap term
scans a whole line per cell. Fine for the target lengths we see in
UniRef50 (< 1000 residues typical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

from plmmsa.align.base import Alignment, AlignMode, MatrixAligner


@dataclass(slots=True)
class _Span:
    """One reportable region inside a single traceback path."""

    score: float  # mean similarity over the span
    length: int
    columns: list[tuple[int, int]]  # (qi, ti) or gap markers
    qi_start: int
    qi_end: int  # exclusive
    ti_start: int
    ti_end: int  # exclusive


class PlmBlast(MatrixAligner):
    """Multi-path SW with moving-average span extraction.

    NOTE on threading: the MatrixAligner base class fans out per-target
    align_matrix calls over a thread pool. pLM-BLAST's JIT kernel plus
    the Python-layer span extraction currently *degrade* under that
    fanout — a 16-target batch went from 12 s at 1 thread to 170 s at
    16 threads in local profiling. Until we identify the contention
    point (numba dispatch lock + allocator thrash are the suspects;
    see PLAN.md), `align()` runs sequentially and ignores the pool
    size. Throughput scaling is via multiple worker containers for now.
    """

    id = "plm_blast"
    display_name = "pLM-BLAST (multi-path SW + span extraction)"
    default_score_matrix = "dot_zscore"

    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Any,  # Sequence[np.ndarray]
        *,
        mode: AlignMode = "local",
        score_matrix: str | None = None,
        normalize: bool | None = None,
        **kwargs: Any,
    ) -> list[Alignment]:
        if mode not in ("local", "global"):
            raise ValueError(
                f"pLM-BLAST supports mode='local'|'global'; got {mode!r}. "
                "Use aligner='otalign' for glocal / q2t / t2q."
            )
        # Sequential override: skip the base class's ThreadPoolExecutor.
        from plmmsa.align.score_matrix import get_builder

        sm = score_matrix or self.default_score_matrix
        if score_matrix is None and normalize is not None:
            sm = "cosine" if normalize else "dot"
        builder = get_builder(sm)
        query = np.asarray(query_embedding, dtype=np.float32)
        targets = [np.asarray(t, dtype=np.float32) for t in target_embeddings]
        sim_matrices = builder.build(query, targets)
        return [self.align_matrix(sim, mode=mode, **kwargs) for sim in sim_matrices]

    # Tunables — all overridable per-request via the aligner options.
    DEFAULT_GAP_PENALTY = 0.0
    DEFAULT_MIN_SPAN = 20
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_SIGMA_FACTOR = 1.0
    DEFAULT_BORDER_STRIDE = 1  # 1 = seed every border cell

    def align_matrix(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        gap_penalty: float | None = None,
        min_span: int | None = None,
        window_size: int | None = None,
        sigma_factor: float | None = None,
        border_stride: int | None = None,
        **_: Any,
    ) -> Alignment:
        spans = self._all_spans(
            sim,
            mode=mode,
            gap_penalty=_opt(gap_penalty, self.DEFAULT_GAP_PENALTY),
            min_span=_opt(min_span, self.DEFAULT_MIN_SPAN),
            window_size=_opt(window_size, self.DEFAULT_WINDOW_SIZE),
            sigma_factor=_opt(sigma_factor, self.DEFAULT_SIGMA_FACTOR),
            border_stride=_opt(border_stride, self.DEFAULT_BORDER_STRIDE),
        )
        if not spans:
            # No significant spans. Return an empty alignment; the caller
            # can filter hits by `score > 0` or `length > 0`.
            return Alignment(
                score=0.0,
                mode=mode,
                query_start=0,
                query_end=0,
                target_start=0,
                target_end=0,
            )
        best = spans[0]  # sorted desc by score
        return Alignment(
            score=best.score,
            mode=mode,
            query_start=best.qi_start,
            query_end=best.qi_end,
            target_start=best.ti_start,
            target_end=best.ti_end,
            columns=list(best.columns),
        )

    def align_matrix_all(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        **kwargs: Any,
    ) -> list[Alignment]:
        """All reportable spans, sorted desc by score. Not part of the
        Aligner contract — exposed here so callers that want the
        multi-path shape can opt in."""
        spans = self._all_spans(
            sim,
            mode=mode,
            gap_penalty=kwargs.get("gap_penalty", self.DEFAULT_GAP_PENALTY),
            min_span=kwargs.get("min_span", self.DEFAULT_MIN_SPAN),
            window_size=kwargs.get("window_size", self.DEFAULT_WINDOW_SIZE),
            sigma_factor=kwargs.get("sigma_factor", self.DEFAULT_SIGMA_FACTOR),
            border_stride=kwargs.get("border_stride", self.DEFAULT_BORDER_STRIDE),
        )
        return [
            Alignment(
                score=s.score,
                mode=mode,
                query_start=s.qi_start,
                query_end=s.qi_end,
                target_start=s.ti_start,
                target_end=s.ti_end,
                columns=list(s.columns),
            )
            for s in spans
        ]

    # --- Core algorithm -----------------------------------------------------

    def _all_spans(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode,
        gap_penalty: float,
        min_span: int,
        window_size: int,
        sigma_factor: float,
        border_stride: int,
    ) -> list[_Span]:
        sim = np.ascontiguousarray(sim, dtype=np.float32)
        lq, lt = sim.shape
        if lq == 0 or lt == 0:
            return []

        H = _fill_dp(sim, gap_penalty=gap_penalty, mode=mode)

        seeds = _border_seeds(H, min_span=min_span, mode=mode, stride=border_stride)

        # Dedup on span start coordinates: a path seeded from a neighboring
        # border cell usually reproduces an adjacent path's high-scoring
        # run. Keep the best-scoring span per `(qi_start, ti_start)`.
        best_by_start: dict[tuple[int, int], _Span] = {}
        # Sigma is measured against the similarity matrix, not H — matches
        # upstream's notion of "cells well above typical noise."
        sigma = max(float(sim.std()), 0.1)
        cutoff = sigma_factor * sigma

        for si, sj in seeds:
            path = _traceback(H, sim, si, sj)
            if len(path) < min_span:
                continue
            for span in _extract_spans(
                path,
                sim,
                window=window_size,
                cutoff=cutoff,
                min_span=min_span,
            ):
                key = (span.qi_start, span.ti_start)
                prior = best_by_start.get(key)
                if prior is None or span.score > prior.score:
                    best_by_start[key] = span

        out = list(best_by_start.values())
        out.sort(key=lambda s: s.score, reverse=True)
        return out


# --- Free functions (easy to unit-test in isolation) ------------------------


def _fill_dp(
    sim: np.ndarray,
    *,
    gap_penalty: float,
    mode: AlignMode,
) -> np.ndarray:
    """Python wrapper — see `_fill_dp_jit` for the hot loop.

    Local mode clamps to zero at each cell; global mode uses no floor and
    seeds H[0,j] / H[i,0] with the linear gap cost. The gap term scans the
    whole preceding column/row per cell — O(Lq * Lt * max(Lq, Lt)) overall.
    numba-JIT on the inner function brings this from ~hours to ~seconds
    on k=500 by 400-residue targets (matches upstream's perf).
    """
    return _fill_dp_jit(
        np.ascontiguousarray(sim, dtype=np.float32),
        np.float32(gap_penalty),
        1 if mode == "local" else 0,
    )


@numba.njit(
    "float32[:, ::1](float32[:, ::1], float32, int32)",
    cache=True,
    nogil=True,
    fastmath=True,
)
def _fill_dp_jit(sim: np.ndarray, g: np.float32, is_local: int) -> np.ndarray:
    """JIT-compiled pLM-BLAST DP fill.

    Same recurrence as the Python reference:
        diag    = H[i-1, j-1] + sim[i-1, j-1]
        col_max = max(H[1:i, j]) - g
        row_max = max(H[i, 1:j]) - g
        H[i, j] = max(diag, col_max, row_max)   # clamped at 0 in local

    `is_local` is a 0/1 int (numba doesn't handle Literal strings in the
    hot path); caller translates from the `mode` enum. `nogil=True` means
    the outer thread pool can truly parallelize across targets without
    the GIL re-serializing things.
    """
    lq, lt = sim.shape
    H = np.zeros((lq + 1, lt + 1), dtype=np.float32)

    if is_local == 0:
        # Global mode: boundary rows pay cumulative linear gap.
        for i in range(1, lq + 1):
            H[i, 0] = -g * i
        for j in range(1, lt + 1):
            H[0, j] = -g * j

    neg_inf = np.float32(-1e30)
    for i in range(1, lq + 1):
        for j in range(1, lt + 1):
            diag = H[i - 1, j - 1] + sim[i - 1, j - 1]
            # Column-max over H[1:i, j]; manual loop avoids slicing that
            # numba would translate into a temporary allocation.
            col_max = neg_inf
            if i > 1:
                for ki in range(1, i):
                    v = H[ki, j]
                    if v > col_max:
                        col_max = v
            row_max = neg_inf
            if j > 1:
                for kj in range(1, j):
                    v = H[i, kj]
                    if v > row_max:
                        row_max = v

            best = diag
            alt = col_max - g
            if alt > best:
                best = alt
            alt = row_max - g
            if alt > best:
                best = alt
            if is_local == 1 and best < np.float32(0.0):
                best = np.float32(0.0)
            H[i, j] = best
    return H


def _border_seeds(
    H: np.ndarray,
    *,
    min_span: int,
    mode: AlignMode,
    stride: int,
) -> list[tuple[int, int]]:
    """Seed traceback from high-scoring border cells.

    In global mode the only seed is the bottom-right corner (standard NW).
    In local mode we walk the right edge and bottom edge, skipping the
    corner region within `min_span` of the origin (can't host a span
    of the required length) and striding to thin out dense starts.
    """
    lq1, lt1 = H.shape  # (lq+1, lt+1)
    if mode == "global":
        return [(lq1 - 1, lt1 - 1)]

    seeds: list[tuple[int, int]] = []
    # Right edge (j = lt). Walk i from top to bottom, skip the near-corner
    # where a useful span couldn't fit.
    for i in range(min_span, lq1, max(1, stride)):
        seeds.append((i, lt1 - 1))
    # Bottom edge (i = lq). Walk j; same cutoff.
    for j in range(min_span, lt1, max(1, stride)):
        seeds.append((lq1 - 1, j))
    # Dedup while preserving order (corner ends up once).
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for s in seeds:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    # Rank seeds by H value descending so the strongest signals drive the
    # dedup below. Stable on equal scores.
    unique.sort(key=lambda ij: float(H[ij[0], ij[1]]), reverse=True)
    return unique


def _traceback(
    H: np.ndarray,
    sim: np.ndarray,
    si: int,
    sj: int,
    stop_value: float = 1e-3,
) -> list[tuple[int, int]]:
    """Walk H backwards from `(si, sj)` following the argmax predecessor.

    Returns a list of `(qi, ti)` columns in alignment order (start → end).
    Gaps are encoded as `(qi, -1)` (gap in target) or `(-1, ti)` (gap in
    query). Stops when H drops below `stop_value` or we hit the origin.
    """
    path: list[tuple[int, int]] = []
    i, j = si, sj
    while i > 0 and j > 0 and float(H[i, j]) >= stop_value:
        # Identify which of (diag, col-jump, row-jump) produced H[i,j].
        diag = float(H[i - 1, j - 1]) + float(sim[i - 1, j - 1])
        col_best_idx = int(np.argmax(H[1:i, j])) + 1 if i > 1 else 0
        row_best_idx = int(np.argmax(H[i, 1:j])) + 1 if j > 1 else 0
        col_best = float(H[col_best_idx, j]) if col_best_idx else -np.inf
        row_best = float(H[i, row_best_idx]) if row_best_idx else -np.inf

        cell = float(H[i, j])
        took_diag = np.isclose(cell, max(diag, 0.0)) and diag >= col_best and diag >= row_best
        if took_diag:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif col_best >= row_best:
            # Gap in target: multiple i-steps collapse into one — emit one
            # gap marker per skipped row so the A3M assembler sees the
            # right length.
            for ri in range(i, col_best_idx, -1):
                path.append((ri - 1, -1))
            i = col_best_idx
        else:
            for rj in range(j, row_best_idx, -1):
                path.append((-1, rj - 1))
            j = row_best_idx

    path.reverse()
    return path


def _extract_spans(
    path: list[tuple[int, int]],
    sim: np.ndarray,
    *,
    window: int,
    cutoff: float,
    min_span: int,
) -> list[_Span]:
    """Segment a traceback path into reportable spans.

    Walks the path's similarity values with a moving-average window and
    emits runs where the smoothed score is above `cutoff`. Gap columns
    contribute 0 to the moving average so gaps down-weight the smoothed
    signal (matches upstream's behavior of penalizing gap-rich stretches
    without an explicit affine penalty).
    """
    if not path:
        return []
    vals = np.asarray(
        [float(sim[qi, ti]) if qi >= 0 and ti >= 0 else 0.0 for qi, ti in path],
        dtype=np.float32,
    )
    if len(vals) < min_span:
        return []
    mean = _moving_average(vals, window=window)
    # Indices where the smoothed signal is above cutoff.
    above = mean >= cutoff

    spans: list[_Span] = []
    n = len(path)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_span:
            sub = path[i:j]
            q_idxs = [qi for qi, _ in sub if qi >= 0]
            t_idxs = [ti for _, ti in sub if ti >= 0]
            if q_idxs and t_idxs:
                span_vals = vals[i:j]
                spans.append(
                    _Span(
                        score=float(span_vals.mean()),
                        length=j - i,
                        columns=list(sub),
                        qi_start=q_idxs[0],
                        qi_end=q_idxs[-1] + 1,
                        ti_start=t_idxs[0],
                        ti_end=t_idxs[-1] + 1,
                    )
                )
        i = j
    return spans


def _moving_average(vals: np.ndarray, *, window: int) -> np.ndarray:
    """Simple uniform moving average, same-length output, edge-truncated.

    `window = 1` returns the input; larger windows smooth over that many
    cells. Ends use a shorter window rather than padding so the smoothed
    signal doesn't dip near the path boundaries.
    """
    if window <= 1 or len(vals) == 0:
        return vals.astype(np.float32, copy=True)
    w = min(window, len(vals))
    out = np.empty_like(vals)
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    for i in range(len(vals)):
        lo = max(0, i - w // 2)
        hi = min(len(vals), lo + w)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def _opt(v: Any, default: Any) -> Any:
    return default if v is None else v


__all__ = ["PlmBlast"]
