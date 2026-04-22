from __future__ import annotations

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from plmmsa.admin.tokens import TokenStore


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, TokenStore]:
    monkeypatch.setenv("ADMIN_TOKEN", "bootstrap-secret")
    from plmmsa.api import app

    store = TokenStore(FakeAsyncRedis())
    app.state.token_store = store
    return TestClient(app), store


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_mint_requires_auth(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.post("/admin/tokens", json={"label": "x"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "E_AUTH_MISSING"


def test_mint_with_bootstrap_token_succeeds(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.post(
            "/admin/tokens",
            headers=_auth("bootstrap-secret"),
            json={"label": "colab-notebook"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"]  # plaintext present
    assert body["record"]["label"] == "colab-notebook"
    assert body["record"]["revoked"] is False


def test_minted_token_is_accepted_on_v2_embed(client) -> None:
    tc, _ = client
    with tc as c:
        mint = c.post(
            "/admin/tokens",
            headers=_auth("bootstrap-secret"),
            json={"label": "test-client"},
        )
        new_token = mint.json()["token"]
        # Using the newly-minted token on /v2/embed: it should pass auth and
        # fail only because the upstream embedding service isn't running.
        resp = c.post(
            "/v2/embed",
            headers=_auth(new_token),
            json={"model": "ankh_cl", "sequences": ["MKT"]},
        )
    # Anything BUT 401/403 means auth passed.
    assert resp.status_code not in {401, 403}, resp.json()


def test_list_tokens(client) -> None:
    tc, _ = client
    with tc as c:
        c.post("/admin/tokens", headers=_auth("bootstrap-secret"), json={"label": "a"})
        c.post("/admin/tokens", headers=_auth("bootstrap-secret"), json={"label": "b"})
        resp = c.get("/admin/tokens", headers=_auth("bootstrap-secret"))
    assert resp.status_code == 200
    labels = [t["label"] for t in resp.json()["tokens"]]
    assert set(labels) == {"a", "b"}


def test_revoke_flips_token_invalid(client) -> None:
    tc, _ = client
    with tc as c:
        mint = c.post(
            "/admin/tokens",
            headers=_auth("bootstrap-secret"),
            json={"label": "revoke-me"},
        )
        minted_token = mint.json()["token"]
        tid = mint.json()["record"]["id"]
        revoke = c.delete(f"/admin/tokens/{tid}", headers=_auth("bootstrap-secret"))
        assert revoke.status_code == 204
        # The revoked token must now fail on protected endpoints.
        follow = c.post(
            "/v2/embed",
            headers=_auth(minted_token),
            json={"model": "ankh_cl", "sequences": ["MKT"]},
        )
    assert follow.status_code == 401
    assert follow.json()["code"] == "E_AUTH_INVALID"


def test_revoke_unknown_404(client) -> None:
    tc, _ = client
    with tc as c:
        resp = c.delete("/admin/tokens/unknown-id", headers=_auth("bootstrap-secret"))
    assert resp.status_code == 404
    assert resp.json()["code"] == "E_JOB_NOT_FOUND"
