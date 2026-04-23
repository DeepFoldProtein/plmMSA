"""Shard store reader coverage.

Builds a tiny on-disk fixture (one `.pt` file in a shard dir, one in a
fallback dir, a fragment-suffixed variant, plus a stale index row pointing
at a non-existent file) and asserts the reader resolves hits, misses,
fragments, and stale rows correctly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from plmmsa.embedding.shard_store import ShardDimMismatch, ShardStore


def _write_tensor(path: Path, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.zeros(shape, dtype=torch.float32)
    torch.save(tensor, path)


def _build_index(db_path: Path, rows: list[tuple[str, str]]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE files (id INTEGER PRIMARY KEY, folder_name TEXT, file_path TEXT)"
        )
        conn.executemany("INSERT INTO files (folder_name, file_path) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def shard_root(tmp_path: Path) -> Path:
    # Shard `200/` holds A.pt and STALE.pt (but we'll only write A.pt).
    _write_tensor(tmp_path / "200" / "A.pt", (5, 4))
    _write_tensor(tmp_path / "300" / "B.pt", (3, 4))
    # Fallback dir with an exact match and a fragment variant.
    _write_tensor(tmp_path / "missing_embeddings" / "C.pt", (2, 4))
    _write_tensor(tmp_path / "missing_embeddings" / "D_F1.pt", (1, 4))
    _write_tensor(tmp_path / "missing_embeddings" / "D_F2.pt", (7, 4))
    # Index with a valid row and a stale row (file doesn't exist).
    _build_index(
        tmp_path / "index.db",
        [
            ("200", "A.pt"),
            ("300", "B.pt"),
            ("999", "STALE.pt"),
        ],
    )
    return tmp_path


def test_index_hit(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, missing = store.fetch(["UniRef50_A", "UniRef50_B"])
    assert sorted(found.keys()) == ["UniRef50_A", "UniRef50_B"]
    assert found["UniRef50_A"].shape == (5, 4)
    assert found["UniRef50_B"].shape == (3, 4)
    assert missing == []


def test_fallback_dir_hit(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, missing = store.fetch(["UniRef50_C"])
    assert "UniRef50_C" in found
    assert found["UniRef50_C"].shape == (2, 4)
    assert missing == []


def test_fragment_variant_resolves_to_highest(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, missing = store.fetch(["UniRef50_D"])
    assert "UniRef50_D" in found
    # D_F2.pt has shape (7, 4); D_F1.pt has (1, 4). Reader should pick F2.
    assert found["UniRef50_D"].shape == (7, 4)
    assert missing == []


def test_stale_index_row_degrades_to_miss(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, missing = store.fetch(["UniRef50_STALE"])
    assert missing == ["UniRef50_STALE"]
    assert found == {}


def test_unknown_id_is_miss(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, missing = store.fetch(["UniRef50_UNKNOWN"])
    assert missing == ["UniRef50_UNKNOWN"]
    assert found == {}


def test_dim_mismatch_raises(tmp_path: Path) -> None:
    # Write a tensor with wrong dim and point the store at dim=1024.
    _write_tensor(tmp_path / "0" / "X.pt", (3, 4))
    _build_index(tmp_path / "index.db", [("0", "X.pt")])
    store = ShardStore(tmp_path, dim=1024)
    with pytest.raises(ShardDimMismatch):
        store.fetch(["UniRef50_X"])


def test_missing_root_returns_all_misses(tmp_path: Path) -> None:
    # Path exists but has no index + no shards.
    store = ShardStore(tmp_path / "does-not-exist", dim=4)
    found, missing = store.fetch(["UniRef50_Q"])
    assert found == {}
    assert missing == ["UniRef50_Q"]


def test_preserves_np_float32(shard_root: Path) -> None:
    store = ShardStore(
        shard_root,
        fallback_dirs=("missing_embeddings",),
        dim=4,
    )
    found, _ = store.fetch(["UniRef50_A"])
    assert found["UniRef50_A"].dtype == np.float32
