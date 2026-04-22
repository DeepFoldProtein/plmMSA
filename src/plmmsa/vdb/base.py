from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Neighbor:
    """Single FAISS hit: identifier + distance.

    Lower distance = closer under L2; higher = closer under inner-product.
    """

    id: str
    distance: float


class VDB(ABC):
    """Vector database collection.

    Concrete subclasses declare `id` (matches settings.toml `[vdb.collections.*]`
    key), `display_name`, `model_backend` (which PLM's vectors populate this
    index), and `dim`, then implement `search`.
    """

    id: str
    display_name: str
    model_backend: str
    dim: int

    @abstractmethod
    def search(
        self,
        vectors: np.ndarray,
        k: int,
        nprobe: int | None = None,
    ) -> list[list[Neighbor]]:
        """Batched k-NN search.

        `vectors` is a float32 `[B, D]` array. Returns one list of `Neighbor`s
        per input row, length `k` (or fewer if the index holds fewer vectors).
        `nprobe` overrides the default for IVF indexes.
        """
        ...
