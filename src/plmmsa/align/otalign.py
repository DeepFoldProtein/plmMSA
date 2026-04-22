from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from plmmsa.align.base import Aligner, Alignment, AlignMode
from plmmsa.errors import ErrorCode, PlmMSAError


class OTalign(Aligner):
    """Optimal-transport alignment (scaffold).

    The interface is declared so the registry + server can list OTalign as a
    known backend. Any actual `align()` call returns 501 `E_NOT_IMPLEMENTED`
    until the real implementation lands.
    """

    id = "otalign"
    display_name = "OTalign (not yet implemented)"

    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        **kwargs: Any,
    ) -> list[Alignment]:
        raise PlmMSAError(
            "OTalign is not yet implemented.",
            code=ErrorCode.NOT_IMPLEMENTED,
            http_status=501,
            detail={"aligner": self.id},
        )
