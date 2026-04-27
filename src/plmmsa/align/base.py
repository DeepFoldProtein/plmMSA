from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

AlignMode = Literal["local", "global", "glocal", "q2t", "t2q"]
# "local"  — SW. Free start/end on both sides, negative-floor clamp.
# "global" — NW. Pay end-gap cost on both sides.
# "glocal" — semi-global. Free end-gaps on both sides.
# "q2t"    — free query end-gap only (template is global).
# "t2q"    — free template end-gap only (query is global).
#
# PLMAlign / pLM-BLAST only understand "local" / "global" today and
# raise on the other three. OTalign accepts all five natively.


@dataclass(slots=True)
class Alignment:
    """A single pairwise alignment between a query and a target.

    `columns` is the alignment path: a list of `(query_idx, target_idx)` pairs
    where a value of `-1` marks a gap in that sequence. The list is ordered
    from the start of the alignment to the end.
    """

    score: float
    mode: AlignMode
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    columns: list[tuple[int, int]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.columns)

    def render(self, query_seq: str, target_seq: str) -> tuple[str, str]:
        """Return the two gap-padded aligned strings (`-` = gap)."""
        q_out: list[str] = []
        t_out: list[str] = []
        for qi, ti in self.columns:
            q_out.append(query_seq[qi] if qi >= 0 else "-")
            t_out.append(target_seq[ti] if ti >= 0 else "-")
        return "".join(q_out), "".join(t_out)

    def identity(self, query_seq: str, target_seq: str) -> float:
        """Fraction of aligned (non-gap) columns where residues match."""
        matches = 0
        total = 0
        for qi, ti in self.columns:
            if qi < 0 or ti < 0:
                continue
            total += 1
            if query_seq[qi] == target_seq[ti]:
                matches += 1
        return matches / total if total else 0.0


class Aligner(ABC):
    """Pairwise aligner over per-residue PLM embeddings.

    The base contract is one method:

        align(query, targets, *, mode, **kw) -> list[Alignment]

    Two flavors of subclass exist because not every algorithm can consume
    a precomputed similarity matrix:

    - **Matrix aligners** (`MatrixAligner`). The algorithm acts on a
      pairwise similarity matrix after it's been built by a
      `ScoreMatrixBuilder`. PLMAlign (affine-gap SW), pLM-BLAST
      (multi-path SW) live here. These subclasses implement
      `align_matrix(sim, ...)` and inherit a default `align()` that
      composes builder + `align_matrix`.
    - **Embedding aligners**. The algorithm needs the raw embeddings —
      e.g. Sinkhorn-based OT where the cost matrix is recomputed (or
      updated) every iteration. These subclasses override `align()`
      directly and never build a precomputed matrix.

    The `align_matrix` method on the base raises `NotImplementedError` by
    default so embedding-aligners don't have to stub it.
    """

    id: str
    display_name: str
    default_score_matrix: str = "dot_zscore"

    @abstractmethod
    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        **kwargs: Any,
    ) -> list[Alignment]:
        """Align `query_embedding` against each target; one Alignment per target."""
        ...

    def align_matrix(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        raw_sim: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Alignment:
        """Optional fast path for matrix-consuming aligners. Default raises
        so embedding-only aligners (Sinkhorn, neural) aren't forced to stub.

        `raw_sim` is the un-normalized similarity matrix (see
        `score_matrix.raw_similarity_for_scoring`) — aligners that follow
        upstream PLMAlign's `mean(raw[path])` scoring read it; legacy
        aligners that score on the DP matrix can ignore it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is an embedding aligner — call align() directly"
        )


class MatrixAligner(Aligner):
    """Aligner that operates on a precomputed `[Lq, Lt]` similarity matrix.

    Subclasses implement `align_matrix` only; the default `align()` below
    composes a `ScoreMatrixBuilder` with per-target dispatch. This keeps
    scoring policy (dot_zscore / cosine / dot / future GPU variants) and
    alignment algorithm (affine-gap SW, pLM-BLAST multi-path SW, ...)
    orthogonal.
    """

    @abstractmethod
    def align_matrix(
        self,
        sim: np.ndarray,
        *,
        mode: AlignMode = "local",
        raw_sim: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Alignment:
        """Align one `[Lq, Lt]` similarity matrix → one Alignment.

        `sim` is the matrix the DP runs over (often Z-scored). `raw_sim`,
        when supplied, is the un-normalized similarity used to score the
        traceback path the upstream-compatible way (`mean(raw[path])`,
        see `score_matrix.raw_similarity_for_scoring`). Aligners that
        don't implement that scoring policy may ignore `raw_sim`.

        Backend-specific tunables (gap_open, gap_extend, ...) ride on
        kwargs; the concrete subclass documents its own options.
        """
        ...

    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        score_matrix: str | None = None,
        normalize: bool | None = None,  # legacy alias: True → cosine, False → dot
        **kwargs: Any,
    ) -> list[Alignment]:
        # Lazy import keeps the base free of matrix-construction details.
        from plmmsa.align.score_matrix import get_builder, raw_similarity_for_scoring

        sm = score_matrix or self.default_score_matrix
        if score_matrix is None and normalize is not None:
            sm = "cosine" if normalize else "dot"

        builder = get_builder(sm)
        query = np.asarray(query_embedding, dtype=np.float32)
        targets = [np.asarray(t, dtype=np.float32) for t in target_embeddings]
        sim_matrices = builder.build(query, targets)
        # Upstream PLMAlign reports `mean(raw[path])` as the alignment score
        # (plmalign_util/alignment.py:204) — DP runs on the (possibly
        # Z-scored) matrix from `build()`, but the score itself is computed
        # from the raw similarity matrix. We materialize that raw matrix
        # once here and thread it through to `align_matrix`. For modes
        # where DP and scoring share a matrix (`dot`, `cosine` today), this
        # is identical to `sim`. Aligners that opt into upstream-compat
        # scoring read `raw_sim`; those that don't can ignore it.
        raw_matrices = [raw_similarity_for_scoring(sm, query, t) for t in targets]

        # Thread-pool fanout across targets. The DP kernels are JIT-
        # compiled with `nogil=True`, so threads actually parallelize
        # instead of being serialized by the GIL. Small batches skip
        # the pool overhead and run sequentially.
        n = len(sim_matrices)
        pool_size = _resolve_pool_size()
        pairs = list(zip(sim_matrices, raw_matrices, strict=True))
        if n <= 1 or pool_size <= 1:
            return [self.align_matrix(sim, mode=mode, raw_sim=raw, **kwargs) for sim, raw in pairs]
        with ThreadPoolExecutor(max_workers=min(pool_size, n)) as ex:
            return list(
                ex.map(
                    lambda sr: self.align_matrix(sr[0], mode=mode, raw_sim=sr[1], **kwargs),
                    pairs,
                )
            )


def _resolve_pool_size() -> int:
    """Pool size for within-job target fanout.

    Precedence (highest first):
      1. `PLMMSA_ALIGN_THREADS` env var (per-process override; useful for
         tests and one-off tuning).
      2. `settings.queue.align_threads` when non-zero (operator default).
      3. `min(os.cpu_count(), 32)` as a safety cap so we don't spawn
         hundreds of threads on machines with many cores.

    Returns 1 to force sequential execution (useful in tests / repros).
    """
    raw = os.environ.get("PLMMSA_ALIGN_THREADS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        from plmmsa.config import get_settings

        configured = int(get_settings().queue.align_threads)
        if configured > 0:
            return configured
    except Exception:
        pass
    return min(32, os.cpu_count() or 1)
