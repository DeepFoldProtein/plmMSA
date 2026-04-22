from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "PLMMSA_SETTINGS_FILE",
    str(_REPO_ROOT / "settings.example.toml"),
)


@pytest.fixture(autouse=True)
def _fresh_token_store():
    """Reset `app.state.token_store` to an empty fakeredis-backed store
    before each test. Tests that need pre-seeded tokens mint them via the
    /admin/tokens routes or directly against the store they grab from the
    app state.
    """
    from fakeredis import FakeAsyncRedis

    from plmmsa.admin.tokens import TokenStore
    from plmmsa.api import app

    app.state.token_store = TokenStore(FakeAsyncRedis())
    yield
