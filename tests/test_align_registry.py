from __future__ import annotations

import pytest

from plmmsa.align import registry
from plmmsa.align.base import Alignment, MatrixAligner
from plmmsa.config import get_settings


class _FakeAligner(MatrixAligner):
    id = "fake"
    display_name = "Fake"

    def align_matrix(self, sim, *, mode="local", **kwargs):
        return Alignment(
            score=0.0,
            mode=mode,
            query_start=0,
            query_end=0,
            target_start=0,
            target_end=0,
        )


def test_loaders_covers_known_aligners() -> None:
    assert set(registry.LOADERS.keys()) == {"plmalign", "plm_blast", "otalign"}


def test_default_settings_load_all_three_aligners() -> None:
    """settings.example.toml now enables plmalign, plm_blast, and otalign."""
    settings = get_settings()
    loaded = registry.load_enabled_aligners(settings)
    assert loaded.keys() == {"plmalign", "plm_blast", "otalign"}


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
