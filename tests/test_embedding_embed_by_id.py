"""`/embed_by_id` endpoint coverage.

Uses an in-memory fake `ShardStore` so the test never touches disk. Key
behaviors verified:
- returns `found` + `missing` for mixed batches
- 400 `E_UNSUPPORTED_MODEL` when the requested model has no shard store
- stays reachable when the corresponding PLM backend is NOT loaded
  (this is the "cache-only" deployment mode the operator asked for)
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from plmmsa.embedding.server import create_app
from plmmsa.embedding.shard_store import ShardStore


class _FakeStore(ShardStore):
    """Override `fetch` to skip sqlite + filesystem."""

    def __init__(self, dim: int, embeddings: dict[str, np.ndarray]) -> None:
        self.dim = dim
        self._embeddings = embeddings

    def fetch(self, ids: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        found = {rid: self._embeddings[rid] for rid in ids if rid in self._embeddings}
        missing = [rid for rid in ids if rid not in self._embeddings]
        return found, missing


def _arr(length: int, dim: int) -> np.ndarray:
    return np.zeros((length, dim), dtype=np.float32)


def test_embed_by_id_mixed_hit_miss() -> None:
    store = _FakeStore(
        dim=4,
        embeddings={"UniRef50_A": _arr(5, 4), "UniRef50_B": _arr(3, 4)},
    )
    app = create_app(
        backends_override={},
        shard_stores_override={"prott5": store},
    )
    with TestClient(app) as client:
        resp = client.post(
            "/embed_by_id",
            json={"model": "prott5", "ids": ["UniRef50_A", "UniRef50_B", "UniRef50_MISS"]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "prott5"
    assert body["dim"] == 4
    assert sorted(body["found"].keys()) == ["UniRef50_A", "UniRef50_B"]
    assert len(body["found"]["UniRef50_A"]) == 5
    assert len(body["found"]["UniRef50_A"][0]) == 4
    assert body["missing"] == ["UniRef50_MISS"]


def test_embed_by_id_unsupported_model_is_400() -> None:
    app = create_app(
        backends_override={},
        shard_stores_override={"prott5": _FakeStore(dim=4, embeddings={})},
    )
    with TestClient(app) as client:
        resp = client.post(
            "/embed_by_id",
            json={"model": "ankh_cl", "ids": ["UniRef50_A"]},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_UNSUPPORTED_MODEL"
    assert body["detail"]["shard_models"] == ["prott5"]


def test_embed_by_id_serves_when_backend_is_disabled() -> None:
    """The whole point of shards being orthogonal to enabled-ness: the
    ProtT5 backend can be absent from `backends_override` (≡ `enabled=false`
    + no load), yet `/embed_by_id` still serves. `/embed` for the same
    model should 400 because the model isn't loaded — that's the contract
    test for the cache-only deployment mode."""
    store = _FakeStore(
        dim=4,
        embeddings={"UniRef50_A": _arr(5, 4)},
    )
    app = create_app(
        backends_override={},  # no PLMs loaded
        shard_stores_override={"prott5": store},
    )
    with TestClient(app) as client:
        # shard hit works
        resp = client.post(
            "/embed_by_id",
            json={"model": "prott5", "ids": ["UniRef50_A"]},
        )
        assert resp.status_code == 200
        assert "UniRef50_A" in resp.json()["found"]

        # /embed for the same model fails — backend isn't loaded
        resp2 = client.post(
            "/embed",
            json={"model": "prott5", "sequences": ["MKT"]},
        )
        assert resp2.status_code == 400
        assert resp2.json()["code"] == "E_UNSUPPORTED_MODEL"


def test_embed_by_id_empty_ids_is_422() -> None:
    """Pydantic enforces `min_length=1` on ids."""
    app = create_app(
        backends_override={},
        shard_stores_override={"prott5": _FakeStore(dim=4, embeddings={})},
    )
    with TestClient(app) as client:
        resp = client.post(
            "/embed_by_id",
            json={"model": "prott5", "ids": []},
        )
    assert resp.status_code == 422


def test_embed_by_id_batch_cap_is_422() -> None:
    """max_length=10_000 guards against OOM-via-JSON blowups. Upped from
    1024 to allow k=1000 x 2-retrieval-model unions without splitting."""
    app = create_app(
        backends_override={},
        shard_stores_override={"prott5": _FakeStore(dim=4, embeddings={})},
    )
    with TestClient(app) as client:
        resp = client.post(
            "/embed_by_id",
            json={"model": "prott5", "ids": ["x"] * 20_000},
        )
    assert resp.status_code == 422


# Silence the noisy fastapi deprecation warnings that live in TestClient's
# lifespan path when asyncio_mode=auto applies to this module.
@pytest.fixture(autouse=True)
def _quiet_warnings():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    yield
