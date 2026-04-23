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

    def __init__(
        self,
        device: str | torch.device = "cuda:1",
        hf_id: str = _HF_ID,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        with torch.cuda.device(self.device):
            model = AutoModel.from_pretrained(hf_id)
            self.model = model.to("cuda", dtype=self.dtype)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            return []

        with torch.cuda.device(self.device):
            inputs = self.tokenizer(
                sequences,
                truncation=False,
                padding=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ).to(self.device)
            out = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        if out.last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state.ndim ({out.last_hidden_state.ndim}) != 3")

        results: list[torch.Tensor] = []
        for seq, emb, attn_mask, special_mask in zip(
            sequences,
            out.last_hidden_state,
            inputs["attention_mask"],
            inputs["special_tokens_mask"],
            strict=True,
        ):
            is_not_special = special_mask == 0
            is_not_pad = attn_mask == 1
            final_mask = is_not_special & is_not_pad
            residue_embeddings = emb[final_mask,:].detach().cpu()
            mask_len = final_mask.sum().item()
            if mask_len != len(seq):
                raise ValueError(
                    f"token/residue length mismatch: seq_len={len(seq)} != {mask_len}, "
                    f"but shape: {tuple(residue_embeddings.shape)}"
                )
            results.append(residue_embeddings)
        return results
