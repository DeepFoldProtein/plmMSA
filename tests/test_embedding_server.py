from __future__ import annotations

from collections.abc import Sequence

import torch
from fastapi.testclient import TestClient

from plmmsa.embedding.server import create_app
from plmmsa.plm.base import PLM


class _FakePLM(PLM):
    id = "fake"
    display_name = "Fake"
    dim = 4
    max_length = 10

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        return [torch.zeros((len(s), self.dim)) for s in sequences]


def test_embedding_health_reports_backends() -> None:
    app = create_app(backends_override={"fake": _FakePLM()})
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["service"] == "embedding"
    assert body["status"] == "ok"
    assert body["models"]["fake"]["loaded"] is True
    assert body["models"]["fake"]["dim"] == 4


def test_embedding_embed_ok() -> None:
    app = create_app(backends_override={"fake": _FakePLM()})
    with TestClient(app) as client:
        resp = client.post("/embed", json={"model": "fake", "sequences": ["AAA", "BB"]})
    assert resp.status_code == 200

    body = resp.json()
    assert body["model"] == "fake"
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 3
    assert len(body["embeddings"][1]) == 2


def test_embedding_embed_unknown_model() -> None:
    app = create_app(backends_override={"fake": _FakePLM()})
    with TestClient(app) as client:
        resp = client.post("/embed", json={"model": "ankh_cl", "sequences": ["AAA"]})
    assert resp.status_code == 400

    body = resp.json()
    assert body["code"] == "E_UNSUPPORTED_MODEL"
    assert body["detail"]["requested"] == "ankh_cl"
    assert "fake" in body["detail"]["available"]
