from __future__ import annotations

from pathlib import Path

from fakeredis import FakeAsyncRedis

from plmmsa.pipeline.fetcher import RedisTargetFetcher
from plmmsa.tools.build_sequence_cache import build, iter_fasta


def test_iter_fasta_handles_multiline(tmp_path: Path) -> None:
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">A descr\nMKT\n>B\nAAA\nGGG\n")
    ids_seqs = list(iter_fasta(fasta))
    assert ids_seqs == [("A", "MKT"), ("B", "AAAGGG")]


async def test_build_writes_keys_and_fetcher_round_trip(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A\nMKT\n>B\nAAA\n>C\nGGG\n")
    redis = FakeAsyncRedis()

    count = await build(fasta_path=fasta, redis=redis, batch_size=2)
    assert count == 3

    fetcher = RedisTargetFetcher(redis)
    out = await fetcher.fetch("ankh_uniref50", ["A", "B", "C", "MISSING"])
    assert out == {"A": "MKT", "B": "AAA", "C": "GGG"}


async def test_build_respects_custom_key_format(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A\nMKT\n")
    redis = FakeAsyncRedis()

    await build(
        fasta_path=fasta,
        redis=redis,
        key_format="seq:{collection}:{id}",
        collection="ankh_uniref50",
    )
    assert await redis.get("seq:ankh_uniref50:A") == b"MKT"


async def test_build_raises_without_redis_source(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A\nMKT\n")

    import pytest

    with pytest.raises(ValueError, match="redis"):
        await build(fasta_path=fasta)
