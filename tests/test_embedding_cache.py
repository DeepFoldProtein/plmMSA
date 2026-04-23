"""Coverage for the optional Redis-backed embedding cache.

When `PLMMSA_EMBEDDING_CACHE_URL` is set, the embedding server serves
cached results for repeat (model, sequence) pairs and only re-encodes
the misses.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from fastapi.testclient import TestClient

from plmmsa.plm.base import PLM


class _CountingPLM(PLM):
    id = "fake"
    display_name = "Fake"
    dim = 4
    max_length = 10

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.encode_calls: list[list[str]] = []

    def encode(self, sequences: Sequence[str]) -> list[torch.Tensor]:
        self.encode_calls.append(list(sequences))
        return [torch.ones((len(s), self.dim)) * len(s) for s in sequences]


def test_embedding_cache_serves_repeat_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # Route cache lookups through fakeredis by overriding Redis.from_url at
    # the call site the embedding server uses.
    from fakeredis import FakeAsyncRedis

    fake = FakeAsyncRedis()

    class _StubRedis:
        @staticmethod
        def from_url(url: str, decode_responses: bool = False):
            return fake

    monkeypatch.setenv("PLMMSA_EMBEDDING_CACHE_URL", "redis://ignored:6379")
    monkeypatch.setattr("redis.asyncio.Redis", _StubRedis)

    from plmmsa.embedding.server import create_app

    plm = _CountingPLM()
    app = create_app(backends_override={"fake": plm})

    with TestClient(app) as client:
        first = client.post("/embed", json={"model": "fake", "sequences": ["AAA", "BB"]})
        assert first.status_code == 200
        assert plm.encode_calls == [["AAA", "BB"]]  # first time: encoded both

        second = client.post("/embed", json={"model": "fake", "sequences": ["AAA", "BB"]})
        assert second.status_code == 200
        # Cache hit should have re-served without re-encoding.
        assert plm.encode_calls == [["AAA", "BB"]]

        mixed = client.post("/embed", json={"model": "fake", "sequences": ["AAA", "CCCC"]})
        assert mixed.status_code == 200
        # Only CCCC was a miss.
        assert plm.encode_calls[-1] == ["CCCC"]
