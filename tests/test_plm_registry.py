from __future__ import annotations

import pytest

from plmmsa.config import get_settings
from plmmsa.plm import registry
from plmmsa.plm.base import PLM


class _FakePLM(PLM):
    id = "fake"
    display_name = "Fake"
    dim = 4
    max_length = 10

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cpu")

    def encode(self, sequences):
        import torch

        return [torch.zeros((len(s), self.dim)) for s in sequences]


def test_loaders_covers_all_four_backends() -> None:
    assert set(registry.LOADERS.keys()) == {"ankh_cl", "ankh_large", "esm1b", "prott5"}


def test_load_enabled_backends_returns_partial_map_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ok(_cfg, _env) -> PLM:
        return _FakePLM()

    def boom(_cfg, _env) -> PLM:
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(
        registry,
        "LOADERS",
        {"ankh_cl": ok, "ankh_large": boom, "esm1b": ok, "prott5": boom},
    )

    settings = get_settings()
    loaded = registry.load_enabled_backends(settings, env={})

    assert set(loaded.keys()) == {"ankh_cl", "esm1b"}


def test_load_enabled_backends_skips_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def recording_loader(name):
        def _inner(_cfg, _env):
            calls.append(name)
            return _FakePLM()

        return _inner

    monkeypatch.setattr(
        registry,
        "LOADERS",
        {n: recording_loader(n) for n in registry.LOADERS},
    )

    settings = get_settings()
    monkeypatch.setattr(settings.models.ankh_large, "enabled", False)
    monkeypatch.setattr(settings.models.prott5, "enabled", False)

    loaded = registry.load_enabled_backends(settings, env={})

    assert set(loaded.keys()) == {"ankh_cl", "esm1b"}
    assert set(calls) == {"ankh_cl", "esm1b"}
