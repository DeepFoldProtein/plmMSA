from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoTokenizer

from plmmsa.plm.base import PLM
from procl.model.ankh import AnkhCL as _AnkhCLBackbone

_TOKENIZER_ID = "ElnaggarLab/ankh-large"


class AnkhCL(PLM):
    """DeepFold Ankh-CL: Ankh-Large backbone with a ConvBert CL head."""

    id = "ankh_cl"
    display_name = "Ankh-CL (DeepFold contrastive fine-tune)"
    max_length = 1022  # min across PLMs enforced at the API edge.

    def __init__(
        self,
        checkpoint: str,
        device: str | torch.device = "cuda:0",
        tokenizer_id: str = _TOKENIZER_ID,
    ) -> None:
        if not checkpoint:
            raise ValueError("AnkhCL requires a checkpoint (local path or HF repo id).")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = _AnkhCLBackbone.from_pretrained(
            checkpoint,
            freeze_base=True,
            is_scratch=False,
        )
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.d_model)

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
        per_token = out.hidden_states  # [B, T, D]
        mask = enc["attention_mask"].bool()

        results: list[torch.Tensor] = []
        for seq, tokens, m in zip(seqs, per_token, mask, strict=True):
            token_count = int(m.sum().item())
            # T5-family tokenizer appends one `</s>` token. Residue count =
            # token_count - 1, and should match len(seq) for Ankh (single-
            # residue tokenization of the amino-acid alphabet).
            residues = tokens[: token_count - 1].detach().cpu()
            if residues.shape[0] != len(seq):
                raise ValueError(
                    f"token/residue length mismatch: seq_len={len(seq)}, "
                    f"got {residues.shape[0]} residue tokens"
                )
            results.append(residues)
        return results
