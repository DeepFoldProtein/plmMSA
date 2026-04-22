from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from plmmsa.vdb.base import VDB, Neighbor

logger = logging.getLogger(__name__)


class FaissVDB(VDB):
    """FAISS-backed collection. Loads the index + id-mapping pickle at init.

    The id mapping can be either a list (index-int → id) or a dict
    (index-int → id); both legacy dumps appear in the Protein-VDB-FAISS tree.
    """

    def __init__(
        self,
        *,
        collection_id: str,
        display_name: str,
        model_backend: str,
        dim: int,
        index_path: Path | str,
        id_mapping_path: Path | str,
        default_nprobe: int = 100,
        normalize: bool = True,
    ) -> None:
        self.id = collection_id
        self.display_name = display_name
        self.model_backend = model_backend
        self.dim = dim
        self.default_nprobe = default_nprobe
        self.normalize = normalize

        index_path = Path(index_path)
        id_mapping_path = Path(id_mapping_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not id_mapping_path.is_file():
            raise FileNotFoundError(f"id mapping not found: {id_mapping_path}")

        logger.info("vdb: loading %s from %s", collection_id, index_path)
        self.index: Any = faiss.read_index(str(index_path))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = default_nprobe

        with id_mapping_path.open("rb") as f:
            raw = pickle.load(f)
        self._id_lookup = _build_id_lookup(raw)

        if self.index.d != dim:
            raise ValueError(
                f"dim mismatch for {collection_id}: settings.dim={dim} but "
                f"FAISS index.d={self.index.d}"
            )

    def search(
        self,
        vectors: np.ndarray,
        k: int,
        nprobe: int | None = None,
    ) -> list[list[Neighbor]]:
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D vectors, got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: got {vectors.shape[1]}, collection expects {self.dim}")

        query = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.normalize:
            faiss.normalize_L2(query)
        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query, k)
        results: list[list[Neighbor]] = []
        for row_d, row_i in zip(distances, indices, strict=True):
            hits = [
                Neighbor(id=self._id_lookup(int(i)), distance=float(d))
                for d, i in zip(row_d, row_i, strict=True)
                if i != -1
            ]
            results.append(hits)
        return results


def _build_id_lookup(raw: Any):
    """Return a callable(int) -> str that handles both list and dict mappings."""
    if isinstance(raw, dict):
        return lambda i: str(raw.get(i, f"unknown:{i}"))
    if isinstance(raw, (list, tuple)):
        return lambda i: str(raw[i]) if 0 <= i < len(raw) else f"unknown:{i}"
    raise TypeError(f"unsupported id mapping type: {type(raw).__name__}")
