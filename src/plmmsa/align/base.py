from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

AlignMode = Literal["local", "global"]


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

    Concrete subclasses declare `id` / `display_name` and implement `align`.
    Keyword arguments (`gap_open`, `gap_extend`, `normalize`, …) are passed
    through `**kwargs` so callers can override backend-specific tunables
    without changing the interface.
    """

    id: str
    display_name: str

    @abstractmethod
    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        **kwargs: Any,
    ) -> list[Alignment]:
        """Align `query_embedding` against each entry in `target_embeddings`.

        Shapes: `query_embedding` is `[Lq, D]`; each target is `[Lt_i, D]`. The
        returned list has one `Alignment` per target, in the same order.
        """
        ...
