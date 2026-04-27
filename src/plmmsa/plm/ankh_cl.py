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
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not checkpoint:
            raise ValueError("AnkhCL requires a checkpoint (local path or HF repo id).")
        self.device = torch.device(device)
        self.dtype = dtype
        # `legacy=True` pins the pre-v4.33 T5 tokenization behavior we
        # calibrated against; v5+'s "new" split adds an extra whitespace
        # normalization step that shifts residue-to-token alignment.
        # Silences the deprecation warning too.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, legacy=True)
        with torch.cuda.device(self.device):
            model = _AnkhCLBackbone.from_pretrained(checkpoint, attention_type="flash")
            self.model = model.to("cuda", dtype=self.dtype)  # pyright: ignore[reportArgumentType]
        self.model.eval()
        self.dim = int(self.model.d_model)

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
            residue_embeddings = emb[final_mask, :].detach().cpu()
            mask_len = final_mask.sum().item()
            if mask_len != len(seq):
                raise ValueError(
                    f"token/residue length mismatch: seq_len={len(seq)} != {mask_len}, "
                    f"but shape: {tuple(residue_embeddings.shape)}"
                )
            results.append(residue_embeddings)
        return results
