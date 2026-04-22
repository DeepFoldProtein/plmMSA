from __future__ import annotations

import pytest

from plmmsa.align import registry
from plmmsa.align.base import Aligner
from plmmsa.config import get_settings


class _FakeAligner(Aligner):
    id = "fake"
    display_name = "Fake"

    def align(self, query_embedding, target_embeddings, *, mode="local", **kwargs):
        return []


def test_loaders_covers_both_aligners() -> None:
    assert set(registry.LOADERS.keys()) == {"plmalign", "otalign"}


def test_default_settings_load_plmalign_only() -> None:
    """settings.example.toml enables plmalign, disables otalign (scaffold)."""
    settings = get_settings()
    loaded = registry.load_enabled_aligners(settings)
    assert "plmalign" in loaded
    assert "otalign" not in loaded


def test_registry_catches_loader_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_cfg):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(
        registry,
        "LOADERS",
        {"plmalign": lambda _: _FakeAligner(), "otalign": boom},
    )
    settings = get_settings()
    monkeypatch.setattr(settings.aligners.otalign, "enabled", True)

    loaded = registry.load_enabled_aligners(settings)
    assert set(loaded.keys()) == {"plmalign"}
