from __future__ import annotations

from functools import partial

import torch
import transformers.models.convbert as c_bert
from torch import nn


class GlobalMaxPooling1D(nn.Module):
    """Global max-pool over the timestep dimension."""

    def __init__(self) -> None:
        super().__init__()
        self._pool = partial(torch.max, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._pool(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    """Global average-pool over the timestep dimension."""

    def __init__(self) -> None:
        super().__init__()
        self._pool = partial(torch.mean, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool(x)


class BaseModule(nn.Module):
    """ConvBert-layer stack shared by downstream heads."""

    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str | None = None,
    ) -> None:
        super().__init__()
        self.model_type = "Transformer"

        cfg = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(cfg) for _ in range(num_layers)]
        )

        if pooling is None:
            self.pooling = None
        elif pooling in {"avg", "mean"}:
            self.pooling = GlobalAvgPooling1D()
        elif pooling == "max":
            self.pooling = GlobalMaxPooling1D()
        else:
            raise ValueError(f"pooling must be 'avg', 'mean', 'max', or None — got {pooling!r}")

    def convbert_forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_encoder:
            x = layer(x, attention_mask)[0]
        return x


class ConvBertForHead(BaseModule):
    """ConvBert-based per-residue head used by AnkhCL."""

    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        self.model_type = "Transformer"

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.convbert_forward(last_hidden_state, attention_mask)
