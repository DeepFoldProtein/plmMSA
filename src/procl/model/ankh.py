from __future__ import annotations

import torch
from torch import nn
from transformers import T5Config, T5EncoderModel, T5PreTrainedModel

from procl.model.head.convbert import ConvBertForHead
from procl.model.output.cloutput import CLPredictionOutput

_HEAD_INPUT_DIM = 1536
_HEAD_HIDDEN_DIM = _HEAD_INPUT_DIM // 2
_HEAD_NUM_HEADS = 8
_HEAD_NUM_HIDDEN_LAYERS = 1
_HEAD_KERNEL_SIZE = 7


class AnkhCL(T5PreTrainedModel):
    """Ankh backbone with a ConvBert contrastive-learning head.

    The base T5 encoder weights come from Ankh-Large; the ConvBert head is
    trained on top for contrastive objectives. At inference time we typically
    load with `freeze_base=True, is_scratch=False` and take the head output.
    """

    def __init__(self, config: T5Config, freeze_base: bool, is_scratch: bool) -> None:
        super().__init__(config)
        self.transformer = T5EncoderModel(config)
        self.freeze_base = freeze_base
        self.d_model = config.d_model

        if freeze_base:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if not is_scratch:
            self.head = ConvBertForHead(
                input_dim=_HEAD_INPUT_DIM,
                nhead=_HEAD_NUM_HEADS,
                hidden_dim=_HEAD_HIDDEN_DIM,
                num_hidden_layers=_HEAD_NUM_HIDDEN_LAYERS,
                kernel_size=_HEAD_KERNEL_SIZE,
                dropout=0.0,
            )

        self.activation = nn.Tanh()

    def add_convbert_for_train(self, dropout: float) -> None:
        """Attach a fresh ConvBert head for training (used when `is_scratch=True`)."""
        self.head = ConvBertForHead(
            input_dim=_HEAD_INPUT_DIM,
            nhead=_HEAD_NUM_HEADS,
            hidden_dim=_HEAD_HIDDEN_DIM,
            num_hidden_layers=_HEAD_NUM_HIDDEN_LAYERS,
            kernel_size=_HEAD_KERNEL_SIZE,
            dropout=dropout,
        )

    def _extract_hidden_state(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.transformer(
            tokens,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=True,
        ).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CLPredictionOutput:
        last_hidden_state = self._extract_hidden_state(input_ids, attention_mask)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())
        last_hidden_state = self.head(last_hidden_state, extended_attention_mask)
        last_hidden_state = self.activation(last_hidden_state)

        return CLPredictionOutput(
            loss=None,
            logits=None,
            hidden_states=last_hidden_state,
            attentions=None,
        )
