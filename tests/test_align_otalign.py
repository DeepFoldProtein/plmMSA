from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align.base import Aligner
from plmmsa.align.otalign import OTalign
from plmmsa.errors import ErrorCode, PlmMSAError


def test_otalign_is_aligner_subclass() -> None:
    assert issubclass(OTalign, Aligner)
    assert OTalign.id == "otalign"


def test_otalign_raises_not_implemented() -> None:
    aligner = OTalign()
    q = np.zeros((4, 8), dtype=np.float32)
    t = np.zeros((5, 8), dtype=np.float32)

    with pytest.raises(PlmMSAError) as exc_info:
        aligner.align(q, [t], mode="local")

    assert exc_info.value.code is ErrorCode.NOT_IMPLEMENTED
    assert exc_info.value.http_status == 501
