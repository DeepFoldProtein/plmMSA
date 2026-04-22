from __future__ import annotations

import time

import pytest
from fakeredis import FakeAsyncRedis

from plmmsa.admin.tokens import TokenStore


@pytest.fixture
async def store() -> TokenStore:
    return TokenStore(FakeAsyncRedis())


async def test_mint_and_verify_roundtrip(store: TokenStore) -> None:
    token, record = await store.mint(label="alice")
    assert token  # non-empty plaintext
    assert record.label == "alice"
    assert record.revoked is False

    verified = await store.verify(token)
    assert verified is not None
    assert verified.id == record.id


async def test_verify_rejects_unknown_token(store: TokenStore) -> None:
    await store.mint(label="alice")
    assert await store.verify("not-a-valid-token") is None


async def test_revoke_invalidates_token(store: TokenStore) -> None:
    token, record = await store.mint(label="alice")
    revoked = await store.revoke(record.id)
    assert revoked is not None
    assert revoked.revoked is True

    assert await store.verify(token) is None

    # The record itself is still retrievable for audit.
    fetched = await store.get(record.id)
    assert fetched is not None and fetched.revoked is True


async def test_revoke_unknown_returns_none(store: TokenStore) -> None:
    assert await store.revoke("no-such-id") is None


async def test_expired_tokens_fail_verify(store: TokenStore) -> None:
    past = time.time() - 60
    token, _ = await store.mint(label="alice", expires_at=past)
    assert await store.verify(token) is None


async def test_list_returns_all_records_sorted(store: TokenStore) -> None:
    _, first = await store.mint(label="alice")
    _, second = await store.mint(label="bob")

    records = await store.list()
    assert [r.id for r in records] == [first.id, second.id]


async def test_plaintext_never_roundtrippable(store: TokenStore) -> None:
    token, _ = await store.mint(label="alice")
    # Sanity: the token plaintext should never be retrievable from the store
    # by id or label. We only expose the hash lookup.
    records = await store.list()
    for record in records:
        dump = record.model_dump_json()
        assert token not in dump
