"""Torch-backed ScoreMatrixBuilder.

Drop-in for the numpy builders in `score_matrix.py`. Registered under the
same ids (`dot_zscore` / `cosine` / `dot`) so switching from CPU to GPU is
an operator setting change, not a client API change.

Device policy:
- Default `device=None` means **inherit from the input tensor** at call
  time. Callers can pass torch tensors on any device (CPU / cuda:0 /
  cuda:1) and the builder runs where they already live — no copy, no
  hardcoded GPU.
- Explicit `device="cuda:0"` or any torch-parseable string pins the
  builder. Useful when inputs are numpy and you want them migrated.
- Numpy inputs default to CPU torch, i.e. the builder never implicitly
  moves to GPU without an explicit `device` argument.

Output is always cpu numpy, matching the `ScoreMatrixBuilder` contract —
the downstream aligner consumes numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from plmmsa.align.score_matrix import SCORE_MATRIX_CHOICES, ScoreMatrixBuilder


class TorchBuilder:
    """Torch-backed score-matrix builder for one of the canonical modes."""

    def __init__(self, mode: str, device: str | None = None) -> None:
        if mode not in SCORE_MATRIX_CHOICES:
            raise ValueError(
                f"mode must be one of {SCORE_MATRIX_CHOICES}, got {mode!r}"
            )
        import torch  # heavy import — gated behind construction

        self.id = mode  # same key as the numpy builder; one replaces the other
        self.mode = mode
        # `None` = inherit from input tensor at build-time; keep as Python
        # string until then so we don't materialize a `torch.device` that
        # might be wrong for the call we end up servicing.
        self._device_override: str | None = device
        _ = torch  # reference so the import isn't marked unused

    def build(
        self,
        query,  # np.ndarray or torch.Tensor
        targets: Sequence,  # np.ndarray or torch.Tensor
    ) -> list[np.ndarray]:
        import torch

        device = self._resolve_device(query, targets)
        with torch.inference_mode():
            q = _to_tensor(query, device=device)
            if self.mode == "cosine":
                q = _l2_normalize(q)
            out: list[np.ndarray] = []
            for t_any in targets:
                t = _to_tensor(t_any, device=device)
                if self.mode == "cosine":
                    t = _l2_normalize(t)
                sim = q @ t.T
                if self.mode == "dot_zscore":
                    sim = _zscore(sim)
                out.append(sim.detach().to(dtype=torch.float32).cpu().numpy())
        return out

    def _resolve_device(self, query, targets):
        """Pick the torch device for this build() call.

        Precedence: constructor override > first torch-tensor input >
        CPU. Stays in Python strings / torch-str form so the same
        TorchBuilder instance can service calls on different devices.
        """
        import torch

        if self._device_override is not None:
            return torch.device(self._device_override)
        for obj in (query, *targets):
            dev = _tensor_device(obj)
            if dev is not None:
                return dev
        return torch.device("cpu")


def _tensor_device(obj):
    """Return torch.device for torch tensors, None for numpy arrays."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.device
    except ImportError:
        pass
    return None


def _to_tensor(obj, *, device):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(obj), dtype=torch.float32, device=device)


def _l2_normalize(x):
    """Per-row L2 normalize. Input/output are torch tensors (deliberately
    untyped here to avoid the heavy torch import at module load)."""
    import torch

    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(norms, min=1e-12)


def _zscore(s):
    """Matrix-level Z-score with the `+1e-3` std floor matching the numpy
    builder. `unbiased=False` matches numpy's default `ddof=0`; torch
    defaults to N-1, which would drift by ~1% on small matrices."""
    mean = s.mean()
    std = s.std(unbiased=False)
    return (s - mean) / (std + 1e-3)


def register_all(device: str | None = None) -> None:
    """Replace the default numpy builders with torch-on-device variants.

    Call once at service startup after confirming torch is available.
    `device=None` keeps device inheritance at build-time. An explicit
    string pins every builder to that device.
    """
    from plmmsa.align.score_matrix import register_builder

    for mode in SCORE_MATRIX_CHOICES:
        register_builder(TorchBuilder(mode, device=device))


__all__ = ["TorchBuilder", "register_all"]


# Type alias so callers can see that `TorchBuilder` satisfies the Protocol.
_check: ScoreMatrixBuilder = TorchBuilder("dot", device="cpu") if False else None  # type: ignore[assignment]
