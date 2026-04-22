from __future__ import annotations

from pathlib import Path

from fakeredis import FakeAsyncRedis

from plmmsa.pipeline.fetcher import RedisTargetFetcher
from plmmsa.tools.build_sequence_cache import build, iter_fasta


def test_iter_fasta_handles_multiline(tmp_path: Path) -> None:
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">A descr\nMKT\n>B\nAAA\nGGG\n")
    ids_seqs = list(iter_fasta(fasta))
    assert ids_seqs == [("A", "MKT", None), ("B", "AAAGGG", None)]


def test_iter_fasta_extracts_uniref50_taxid(tmp_path: Path) -> None:
    fasta = tmp_path / "uniref.fasta"
    fasta.write_text(
        ">UniRef50_A0A5A9P0L4 Protein P (Fragment) n=3 Tax=Amphipoda TaxID=1214932 "
        "RepID=A0A5A9P0L4_9CRUS\nMKT\n"
        ">UniRef50_PLAIN no taxid here\nAAA\n"
    )
    rows = list(iter_fasta(fasta))
    assert rows == [
        ("UniRef50_A0A5A9P0L4", "MKT", "1214932"),
        ("UniRef50_PLAIN", "AAA", None),
    ]


async def test_build_writes_seq_and_tax_keys_round_trip(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(
        ">UniRef50_A TaxID=7227\nMKT\n>UniRef50_B TaxID=9606\nAAA\n>UniRef50_C no-tax\nGGG\n"
    )
    redis = FakeAsyncRedis()

    seq_count, tax_count = await build(fasta_path=fasta, redis=redis, batch_size=2)
    assert (seq_count, tax_count) == (3, 2)

    fetcher = RedisTargetFetcher(redis)
    seqs = await fetcher.fetch("ankh_uniref50", ["UniRef50_A", "UniRef50_B", "UniRef50_C", "MISS"])
    assert seqs == {"UniRef50_A": "MKT", "UniRef50_B": "AAA", "UniRef50_C": "GGG"}

    assert await redis.get("tax:UniRef50_A") == b"7227"
    assert await redis.get("tax:UniRef50_B") == b"9606"
    assert await redis.get("tax:UniRef50_C") is None  # no TaxID in header


async def test_build_respects_custom_key_format(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A TaxID=42\nMKT\n")
    redis = FakeAsyncRedis()

    await build(
        fasta_path=fasta,
        redis=redis,
        key_format="seq:{collection}:{id}",
        tax_key_format="tax:{collection}:{id}",
        collection="ankh_uniref50",
    )
    assert await redis.get("seq:ankh_uniref50:A") == b"MKT"
    assert await redis.get("tax:ankh_uniref50:A") == b"42"


async def test_build_tax_suppressed(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A TaxID=42\nMKT\n")
    redis = FakeAsyncRedis()

    seq_count, tax_count = await build(
        fasta_path=fasta,
        redis=redis,
        tax_key_format=None,
    )
    assert (seq_count, tax_count) == (1, 0)
    assert await redis.get("seq:A") == b"MKT"
    assert await redis.get("tax:A") is None


async def test_build_raises_without_redis_source(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A\nMKT\n")

    import pytest

    with pytest.raises(ValueError, match="redis"):
        await build(fasta_path=fasta)
