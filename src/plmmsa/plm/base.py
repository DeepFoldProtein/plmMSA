from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


def dtype_from_precision(precision: str) -> torch.dtype:
    """Map the settings string to a torch dtype.

    Central so every PLM loader resolves precision the same way, and so
    `settings.toml` has a single documented set of accepted values.
    """
    key = precision.lower().strip()
    mapping: dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if key not in mapping:
        raise ValueError(f"unsupported precision {precision!r}; expected one of {sorted(mapping)}")
    return mapping[key]


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
