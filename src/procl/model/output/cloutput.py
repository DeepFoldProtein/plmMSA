from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput


@dataclass
class CLPredictionOutput(ModelOutput):
    """Model output for the contrastive-learning Ankh head.

    `last_hidden_state` holds the per-residue embedding after the ConvBert head;
    `loss`, `logits`, and `attentions` are retained for compatibility with
    training code paths and may be None at inference time.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    attentions: torch.FloatTensor | None = None
