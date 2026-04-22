from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from plmmsa.plm.base import PLM

_HF_ID = "facebook/esm1b_t33_650M_UR50S"


class ESM1b(PLM):
    """Meta ESM-1b (facebook/esm1b_t33_650M_UR50S)."""

    id = "esm1b"
    display_name = "ESM-1b"
    max_length = 1022

    def __init__(self, device: str | torch.device = "cuda:1", hf_id: str = _HF_ID) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id)
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

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
            # ESM adds <cls> at position 0 and <eos> at position token_count-1.
            residues = tokens[1 : token_count - 1].detach().cpu()
            if residues.shape[0] != len(seq):
                raise ValueError(
                    f"token/residue length mismatch for ESM-1b: seq_len={len(seq)}, "
                    f"got {residues.shape[0]} residue tokens"
                )
            results.append(residues)
        return results
