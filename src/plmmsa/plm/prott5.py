from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, T5EncoderModel

from plmmsa.plm.base import PLM

_HF_ID = "Rostlab/prot_t5_xl_uniref50"
_RARE_AA_REMAP = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})


class ProtT5(PLM):
    """Rostlab ProtT5-XL-UniRef50 T5 encoder."""

    id = "prott5"
    display_name = "ProtT5-XL-UniRef50"
    max_length = 1022

    def __init__(self, device: str | torch.device = "cuda:1", hf_id: str = _HF_ID) -> None:
        self.device = torch.device(device)
        # Rostlab's ProtT5 ships a SentencePiece vocab that `transformers`
        # can't convert to its Unigram fast-tokenizer. Force the slow path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, do_lower_case=False, use_fast=False
        )
        model = T5EncoderModel.from_pretrained(hf_id)
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.config.d_model)

    @staticmethod
    def _format(seq: str) -> str:
        return " ".join(seq.upper().translate(_RARE_AA_REMAP))

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        seqs = list(sequences)
        if not seqs:
            return []

        formatted = [self._format(s) for s in seqs]
        enc = self.tokenizer(
            formatted,
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
                    f"token/residue length mismatch for ProtT5: seq_len={len(seq)}, "
                    f"got {residues.shape[0]} residue tokens"
                )
            results.append(residues)
        return results
