from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, T5EncoderModel

from plmmsa.plm.base import PLM

_HF_ID = "ElnaggarLab/ankh-large"


class AnkhLarge(PLM):
    """Plain Ankh-Large T5 encoder (no contrastive head)."""

    id = "ankh_large"
    display_name = "Ankh-Large"
    max_length = 1022

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        hf_id: str = _HF_ID,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, legacy=True)
        model = T5EncoderModel.from_pretrained(hf_id)
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]
        if dtype != torch.float32:
            self.model = self.model.to(dtype)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.config.d_model)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        seqs = list(sequences)
        if not seqs:
            return []

        enc = self.tokenizer(
            seqs,
            truncation=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        per_token = out.last_hidden_state
        mask = enc["attention_mask"].bool()

        results: list[torch.Tensor] = []
        for seq, tokens, m in zip(seqs, per_token, mask, strict=True):
            token_count = int(m.sum().item())
            residues = tokens[: token_count - 1].detach().cpu()
            if residues.shape[0] != len(seq):
                raise ValueError(
                    f"token/residue length mismatch for Ankh-Large: seq_len={len(seq)}, "
                    f"got {residues.shape[0]} residue tokens"
                )
            results.append(residues)
        return results
