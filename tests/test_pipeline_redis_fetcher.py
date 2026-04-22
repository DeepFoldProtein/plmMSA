from __future__ import annotations

from fakeredis import FakeAsyncRedis

from plmmsa.pipeline.fetcher import RedisTargetFetcher


async def test_redis_fetcher_returns_matching_keys() -> None:
    redis = FakeAsyncRedis()
    await redis.set("seq:A", b"MKT")
    await redis.set("seq:B", b"AAAG")

    fetcher = RedisTargetFetcher(redis)
    result = await fetcher.fetch("ankh_uniref50", ["A", "B", "MISSING"])

    assert result == {"A": "MKT", "B": "AAAG"}


async def test_redis_fetcher_honors_custom_key_format() -> None:
    redis = FakeAsyncRedis()
    await redis.set("seq:ankh_uniref50:X", b"MKA")

    fetcher = RedisTargetFetcher(redis, key_format="seq:{collection}:{id}")
    result = await fetcher.fetch("ankh_uniref50", ["X", "Y"])

    assert result == {"X": "MKA"}


async def test_redis_fetcher_empty_ids_short_circuits() -> None:
    redis = FakeAsyncRedis()
    fetcher = RedisTargetFetcher(redis)
    assert await fetcher.fetch("ankh_uniref50", []) == {}
