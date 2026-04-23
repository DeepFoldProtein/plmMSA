from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoTokenizer  # , T5EncoderModel
from turbot5 import T5EncoderForMaskedLM

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
        with torch.cuda.device(self.device):
            model = T5EncoderForMaskedLM.from_pretrained(hf_id, attention_type="flash")
            self.model = model.to("cuda", dtype=self.dtype)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.config.d_model)

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
