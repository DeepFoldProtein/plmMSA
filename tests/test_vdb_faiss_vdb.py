from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import pytest

from plmmsa.vdb.faiss_vdb import FaissVDB

_DIM = 16
_N_VECTORS = 64
_RNG = np.random.default_rng(0)


def _build_synthetic_index(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    vectors = _RNG.standard_normal((_N_VECTORS, _DIM)).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatL2(_DIM)
    index.add(vectors)  # pyright: ignore[reportCallIssue]

    index_path = tmp_path / "tiny.faiss"
    faiss.write_index(index, str(index_path))

    ids = [f"UR50_{i:04d}" for i in range(_N_VECTORS)]
    id_map_path = tmp_path / "tiny.faiss_id_mapping.pkl"
    with id_map_path.open("wb") as f:
        pickle.dump(ids, f)

    return index_path, id_map_path, ids


def _make_vdb(tmp_path: Path) -> tuple[FaissVDB, list[str]]:
    index_path, id_map_path, ids = _build_synthetic_index(tmp_path)
    vdb = FaissVDB(
        collection_id="tiny",
        display_name="Tiny",
        model_backend="ankh_cl",
        dim=_DIM,
        index_path=index_path,
        id_mapping_path=id_map_path,
    )
    return vdb, ids


def test_faiss_vdb_search_returns_k_hits(tmp_path: Path) -> None:
    vdb, ids = _make_vdb(tmp_path)
    query = _RNG.standard_normal((1, _DIM)).astype(np.float32)
    hits = vdb.search(query, k=5)

    assert len(hits) == 1
    assert len(hits[0]) == 5
    for n in hits[0]:
        assert n.id in ids
        assert n.distance >= 0.0


def test_faiss_vdb_search_batched(tmp_path: Path) -> None:
    vdb, _ = _make_vdb(tmp_path)
    queries = _RNG.standard_normal((3, _DIM)).astype(np.float32)
    hits = vdb.search(queries, k=7)

    assert len(hits) == 3
    assert all(len(row) == 7 for row in hits)


def test_faiss_vdb_rejects_wrong_dim(tmp_path: Path) -> None:
    vdb, _ = _make_vdb(tmp_path)
    bad = _RNG.standard_normal((1, _DIM + 1)).astype(np.float32)
    with pytest.raises(ValueError, match="dim mismatch"):
        vdb.search(bad, k=3)


def test_faiss_vdb_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FAISS index"):
        FaissVDB(
            collection_id="missing",
            display_name="M",
            model_backend="ankh_cl",
            dim=_DIM,
            index_path=tmp_path / "does_not_exist.faiss",
            id_mapping_path=tmp_path / "does_not_exist.pkl",
        )
