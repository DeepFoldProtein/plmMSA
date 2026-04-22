from __future__ import annotations

import os

import pytest


def test_ankh_cl_is_plm_subclass() -> None:
    from plmmsa.plm.ankh_cl import AnkhCL
    from plmmsa.plm.base import PLM

    assert issubclass(AnkhCL, PLM)
    assert AnkhCL.id == "ankh_cl"
    assert AnkhCL.max_length == 1022


def test_ankh_cl_rejects_empty_checkpoint() -> None:
    from plmmsa.plm.ankh_cl import AnkhCL

    with pytest.raises(ValueError, match="checkpoint"):
        AnkhCL(checkpoint="", device="cpu")


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Slow: downloads the Ankh-Large tokenizer + Ankh-CL checkpoint (multi-GB). "
    "Set RUN_SLOW=1 to enable.",
)
def test_ankh_cl_encode_roundtrip() -> None:
    import torch

    from plmmsa.plm.ankh_cl import AnkhCL

    checkpoint = os.environ.get("ANKH_CL_CHECKPOINT") or "DeepFoldProtein/Ankh-Large-Contrastive"
    device = os.environ.get("PLMMSA_TEST_DEVICE") or (
        "cuda:1" if torch.cuda.is_available() else "cpu"
    )

    model = AnkhCL(checkpoint=checkpoint, device=device)
    seqs = ["MKTIIALSYIFCLVFA", "MASLKV"]
    out = model.encode(seqs)

    assert len(out) == 2
    for tensor, seq in zip(out, seqs, strict=True):
        assert tensor.device.type == "cpu"
        assert tensor.shape == (len(seq), model.dim)
