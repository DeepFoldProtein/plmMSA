"""Coverage for the Redis-backed MSA result cache.

Shape under test:

- **Canonicalization** — equivalent submits collapse to the same key;
  non-output-affecting fields (`force_recompute`) don't change it.
- **api submit hot path** — on cache hit, POST /v2/msa returns a
  `succeeded` job without touching the work queue; the result payload
  matches the cached one and carries `stats.cache_hit = True`.
- **api opt-out** — `force_recompute=true` bypasses the cache even on a
  hit, falling through to normal enqueue.
- **worker write** — on successful pipeline run the worker populates the
  cache so the next submission hits.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from plmmsa.jobs import JobResult, JobStore, ResultCache, cache_key
from plmmsa.jobs.models import JobStatus

_BOOTSTRAP = "bootstrap-secret"
_AUTH = {"Authorization": f"Bearer {_BOOTSTRAP}"}


# ---------------------------------------------------------------------------
# Canonicalization / key derivation
# ---------------------------------------------------------------------------


def test_cache_key_normalizes_sequence_case_and_whitespace() -> None:
    """A paste with lowercase + line breaks must hash to the same key
    as the cleaned-up equivalent."""
    dirty = {"sequences": ["mkt  iial\n  "], "models": ["ankh_cl"]}
    clean = {"sequences": ["MKTIIAL"], "models": ["ankh_cl"]}
    assert cache_key(dirty) == cache_key(clean)


def test_cache_key_ignores_force_recompute() -> None:
    """`force_recompute` opts out of *consulting* the cache; it must
    not change the *key* or we'd pollute the keyspace every time a
    client bursts with the opt-out set."""
    base = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    with_flag = {**base, "force_recompute": True}
    assert cache_key(base) == cache_key(with_flag)


def test_cache_key_is_sensitive_to_paired_and_score_model() -> None:
    """Anything that changes the MSA output changes the key."""
    base = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    paired = {**base, "paired": True}
    scored = {**base, "score_model": "prott5"}
    assert cache_key(base) != cache_key(paired)
    assert cache_key(base) != cache_key(scored)
    assert cache_key(paired) != cache_key(scored)


def test_cache_key_preserves_chain_order_for_paired() -> None:
    """Chain A vs chain B isn't symmetric — swapping chains produces a
    different paired MSA, so the cache must treat them as distinct."""
    forward = {"sequences": ["MKT", "GGG"], "paired": True}
    reverse = {"sequences": ["GGG", "MKT"], "paired": True}
    assert cache_key(forward) != cache_key(reverse)


def test_cache_key_ignores_empty_collections_and_none_fields() -> None:
    """Missing vs explicit-empty should collide so sibling clients
    (one that passes `collections={}` and one that omits it) don't
    split the keyspace."""
    bare = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    explicit_empty = {**bare, "collections": {}, "options": None}
    assert cache_key(bare) == cache_key(explicit_empty)


async def test_result_cache_get_returns_none_when_disabled() -> None:
    """A `ResultCache(None)` is the degraded/disabled path — every get
    should return None and every set should be a no-op, so callers can
    run unchanged when the operator hasn't configured a cache-emb."""
    cache = ResultCache(None, ttl_s=3600)
    assert cache.enabled is False
    assert await cache.get({"sequences": ["MKT"]}) is None
    # Should not raise:
    await cache.set({"sequences": ["MKT"]}, JobResult(payload=">q\nMKT\n"))


async def test_result_cache_round_trip() -> None:
    redis = FakeAsyncRedis()
    cache = ResultCache(redis, ttl_s=3600)
    payload = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    assert await cache.get(payload) is None

    await cache.set(payload, JobResult(payload=">query\nMKT\n", stats={"depth": 1}))
    got = await cache.get(payload)
    assert got is not None
    assert got.payload == ">query\nMKT\n"
    assert got.stats == {"depth": 1}

    # Canonicalization equivalence: whitespace-dirty input hits the same
    # slot the clean input wrote to.
    dirty = {"sequences": ["mkt\n"], "models": ["ankh_cl"]}
    again = await cache.get(dirty)
    assert again is not None
    assert again.payload == ">query\nMKT\n"


# ---------------------------------------------------------------------------
# api submit hot path — cache hit + opt-out
# ---------------------------------------------------------------------------


@pytest.fixture
def api_with_cache(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, JobStore, ResultCache]:
    monkeypatch.setenv("ADMIN_TOKEN", _BOOTSTRAP)
    store = JobStore(FakeAsyncRedis())
    cache_redis = FakeAsyncRedis()
    cache = ResultCache(cache_redis, ttl_s=3600)

    import plmmsa.api.routes.v2 as v2_mod

    async def fake_get_store() -> JobStore:
        return store

    async def fake_get_cache() -> ResultCache:
        return cache

    monkeypatch.setattr(v2_mod, "_get_job_store", fake_get_store)
    monkeypatch.setattr(v2_mod, "_get_result_cache", fake_get_cache)

    from plmmsa.api import app

    return TestClient(app), store, cache


def _seed_cache_from_first_submit(
    tc: TestClient, store: JobStore, cache: ResultCache, body: dict[str, Any], result: JobResult
) -> None:
    """Drive one submit to capture the exact payload the api stamps
    onto the job, then seed the cache under that same payload. This
    way the test doesn't have to reimplement the api's stamping logic
    (resolved models + score_model + query_ids + collections).

    After seeding, the seeding submit's queue entry, job record, and
    idempotency key are all wiped so the second submit exercises a
    clean cache-hit path (not the idempotency shortcut that would
    otherwise return the same queued job)."""
    with tc as c:
        first = c.post("/v2/msa", json=body).json()
    first_id = first["job_id"]
    job = asyncio.run(store.get(first_id))
    assert job is not None
    asyncio.run(cache.set(job.request, result))
    # FakeAsyncRedis is per-test so flushdb is safe — wipes queue, the
    # seeding job record, and the idempotency key in one shot.
    asyncio.run(store.redis.flushdb())


def test_submit_serves_succeeded_on_cache_hit(api_with_cache) -> None:
    tc, store, cache = api_with_cache
    seeded = JobResult(
        format="a3m",
        payload=">query\nMKTIIAL\n>hit1\nMKTIIAL\n",
        stats={"depth": 2, "hits_fetched": 1},
    )
    submit_body = {"sequences": ["MKTIIAL"], "models": ["ankh_cl"]}
    _seed_cache_from_first_submit(tc, store, cache, submit_body, seeded)

    with tc as c:
        resp = c.post("/v2/msa", json=submit_body)

    assert resp.status_code == 202, resp.json()
    body = resp.json()
    assert body["status"] == "succeeded"
    job_id = body["job_id"]

    with tc as c:
        fetched = c.get(f"/v2/msa/{job_id}").json()
    assert fetched["status"] == "succeeded"
    assert fetched["result"]["payload"] == seeded.payload
    assert fetched["result"]["stats"]["cache_hit"] is True
    assert fetched["result"]["stats"]["depth"] == 2  # existing stats preserved

    # Queue must be empty — a cache hit never enqueues.
    queue_depth = asyncio.run(store.redis.llen(store.queue_key))
    assert queue_depth == 0


def test_submit_force_recompute_bypasses_cache(api_with_cache) -> None:
    tc, store, cache = api_with_cache
    submit_body = {"sequences": ["MKTIIAL"], "models": ["ankh_cl"]}
    _seed_cache_from_first_submit(
        tc, store, cache, submit_body, JobResult(payload=">cached\nMKT\n")
    )

    with tc as c:
        resp = c.post(
            "/v2/msa",
            json={**submit_body, "force_recompute": True},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"  # NOT succeeded — bypass worked

    queue_depth = asyncio.run(store.redis.llen(store.queue_key))
    assert queue_depth == 1


# ---------------------------------------------------------------------------
# Worker writes on success
# ---------------------------------------------------------------------------


class _StubOrchestrator:
    """Minimal orchestrator stand-in — drives `_one_iteration` without
    running the real pipeline."""

    def __init__(self, result: JobResult) -> None:
        self._result = result
        self.calls = 0

    async def run(self, request: dict[str, Any]) -> JobResult:
        self.calls += 1
        return self._result


async def test_worker_writes_result_cache_on_success() -> None:
    from plmmsa.worker.__main__ import _one_iteration

    store = JobStore(FakeAsyncRedis())
    cache_redis = FakeAsyncRedis()
    cache = ResultCache(cache_redis, ttl_s=3600)

    payload = {"sequences": ["MKT"], "models": ["ankh_cl"]}
    job = await store.create(payload)
    result = JobResult(format="a3m", payload=">q\nMKT\n", stats={"depth": 1})
    orchestrator = _StubOrchestrator(result)

    advanced = await _one_iteration(store, orchestrator, cache)  # type: ignore[arg-type]
    assert advanced is True
    assert orchestrator.calls == 1

    refreshed = await store.get(job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.SUCCEEDED

    # Cache should now hold the result under the canonical key.
    cached = await cache.get(payload)
    assert cached is not None
    assert cached.payload == ">q\nMKT\n"


async def test_worker_cache_write_failure_does_not_fail_job() -> None:
    """A Redis outage on the cache must not cause a job failure — the
    result is already persisted via mark_succeeded; caching is best-effort."""
    from plmmsa.worker.__main__ import _one_iteration

    store = JobStore(FakeAsyncRedis())

    class _BrokenRedis:
        async def set(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("cache-emb unreachable")

        async def get(self, *_a: Any, **_kw: Any) -> None:
            raise RuntimeError("cache-emb unreachable")

    cache = ResultCache(_BrokenRedis(), ttl_s=3600)  # type: ignore[arg-type]

    payload = {"sequences": ["MKT"]}
    job = await store.create(payload)
    orchestrator = _StubOrchestrator(JobResult(payload=">q\nMKT\n"))

    advanced = await _one_iteration(store, orchestrator, cache)  # type: ignore[arg-type]
    assert advanced is True
    refreshed = await store.get(job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.SUCCEEDED  # still succeeds


async def test_job_store_insert_terminal_refuses_non_terminal() -> None:
    """Guard against api wiring a bug where it tries to drop a non-terminal
    synthesized job into the store and skip the queue."""
    store = JobStore(FakeAsyncRedis())
    import pytest

    from plmmsa.jobs.models import Job

    queued = Job(id="not-terminal", status=JobStatus.QUEUED, request={}, created_at=0.0)
    with pytest.raises(ValueError, match="terminal"):
        await store.insert_terminal(queued)
