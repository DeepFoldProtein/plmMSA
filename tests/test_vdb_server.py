from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from plmmsa.vdb.base import VDB, Neighbor
from plmmsa.vdb.server import create_app


class _FakeVDB(VDB):
    id = "fake"
    display_name = "Fake"
    model_backend = "ankh_cl"
    dim = 4

    def search(
        self,
        vectors: np.ndarray,
        k: int,
        nprobe: int | None = None,
    ) -> list[list[Neighbor]]:
        return [
            [Neighbor(id=f"hit_{i}", distance=float(i)) for i in range(min(k, 3))]
            for _ in range(vectors.shape[0])
        ]


def test_vdb_health() -> None:
    app = create_app(collections_override={"fake": _FakeVDB()})
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "vdb"
    assert body["collections"]["fake"]["dim"] == 4
    assert body["collections"]["fake"]["model_backend"] == "ankh_cl"


def test_vdb_search_ok() -> None:
    app = create_app(collections_override={"fake": _FakeVDB()})
    with TestClient(app) as client:
        resp = client.post(
            "/search",
            json={
                "collection": "fake",
                "vectors": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                "k": 2,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "fake"
    assert body["k"] == 2
    assert len(body["results"]) == 2
    assert len(body["results"][0]) == 2
    assert body["results"][0][0]["id"] == "hit_0"


def test_vdb_search_unknown_collection() -> None:
    app = create_app(collections_override={"fake": _FakeVDB()})
    with TestClient(app) as client:
        resp = client.post(
            "/search",
            json={"collection": "missing", "vectors": [[0.0, 0.0, 0.0, 0.0]], "k": 1},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_UNSUPPORTED_MODEL"
    assert body["detail"]["requested"] == "missing"
