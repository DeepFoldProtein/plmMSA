from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class PLM(ABC):
    """Per-residue protein language model embedding backend.

    Concrete subclasses declare their `id` (matches the key in settings.toml
    `[models.*]`), `display_name`, `dim`, and `max_length`, and implement
    `encode`. `device` is set at construction time from the backend's own
    device env var.
    """

    id: str
    display_name: str
    dim: int
    max_length: int
    device: torch.device

    @abstractmethod
    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        """Return one `[L_i, D]` CPU tensor per input sequence.

        `L_i` is the residue count of the i-th input (no special tokens).
        `D` equals `self.dim`. The output list preserves input order.
        """
        ...
