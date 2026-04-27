from __future__ import annotations

from pathlib import Path

from fakeredis import FakeAsyncRedis

from plmmsa.pipeline.fetcher import RedisTargetFetcher
from plmmsa.tools.build_sequence_cache import (
    build,
    build_from_fasta,
    iter_csv,
    iter_csv_dir,
    iter_fasta,
)


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


def test_iter_csv_reads_accession_sequence_and_taxid(tmp_path: Path) -> None:
    # Mirrors the UniRef50 split layout at
    # /gpfs/database/milvus/datasets/uniref50_t5/split/*.csv — columns
    # accession, description, sequence, length, length_group. TaxID lives
    # inside description, matching FASTA-header shape.
    csv_path = tmp_path / "0-0.csv"
    csv_path.write_text(
        "accession,description,sequence,length,length_group\n"
        "UPI0001,UniRef50_UPI0001 foo Tax=Bar TaxID=9606 RepID=UPI0001,MKT,3,0\n"
        "UPI0002,UniRef50_UPI0002 no taxid,AAA,3,0\n"
        ",should-be-skipped-blank-acc,AAA,3,0\n"
        "UPI0003,skip-blank-seq,,0,0\n"
    )
    rows = list(iter_csv(csv_path))
    assert rows == [
        ("UPI0001", "MKT", "9606"),
        ("UPI0002", "AAA", None),
    ]


def test_iter_csv_dir_glob_and_slice(tmp_path: Path) -> None:
    header = "accession,description,sequence,length,length_group\n"
    (tmp_path / "0-0.csv").write_text(header + "A,TaxID=1,MKT,3,0\n")
    (tmp_path / "0-1.csv").write_text(header + "B,TaxID=2,AAA,3,0\n")
    (tmp_path / "0-2.csv").write_text(header + "C,TaxID=3,GGG,3,0\n")

    all_rows = list(iter_csv_dir(tmp_path))
    assert [r[0] for r in all_rows] == ["A", "B", "C"]

    sliced = list(iter_csv_dir(tmp_path, start=1, stop=3))
    assert [r[0] for r in sliced] == ["B", "C"]


async def test_build_writes_seq_and_tax_keys_round_trip(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(
        ">UniRef50_A TaxID=7227\nMKT\n>UniRef50_B TaxID=9606\nAAA\n>UniRef50_C no-tax\nGGG\n"
    )
    redis = FakeAsyncRedis()

    seq_count, tax_count = await build(records=iter_fasta(fasta), redis=redis, batch_size=2)
    assert (seq_count, tax_count) == (3, 2)

    fetcher = RedisTargetFetcher(redis)
    seqs = await fetcher.fetch("ankh_uniref50", ["UniRef50_A", "UniRef50_B", "UniRef50_C", "MISS"])
    assert seqs == {"UniRef50_A": "MKT", "UniRef50_B": "AAA", "UniRef50_C": "GGG"}

    assert await redis.get("tax:UniRef50_A") == b"7227"
    assert await redis.get("tax:UniRef50_B") == b"9606"
    assert await redis.get("tax:UniRef50_C") is None  # no TaxID in header


async def test_build_from_csv_dir_round_trip(tmp_path: Path) -> None:
    header = "accession,description,sequence,length,length_group\n"
    (tmp_path / "0-0.csv").write_text(
        header + "UPI0001,UniRef50_UPI0001 TaxID=9606 RepID=UPI0001,MKT,3,0\n"
    )
    (tmp_path / "0-1.csv").write_text(header + "UPI0002,UniRef50_UPI0002 no-taxid,AAAGGG,6,0\n")
    redis = FakeAsyncRedis()

    seq_count, tax_count = await build(
        records=iter_csv_dir(tmp_path),
        redis=redis,
        key_format="seq:UniRef50_{id}",
        tax_key_format="tax:UniRef50_{id}",
        batch_size=1,
    )
    assert (seq_count, tax_count) == (2, 1)
    assert await redis.get("seq:UniRef50_UPI0001") == b"MKT"
    assert await redis.get("seq:UniRef50_UPI0002") == b"AAAGGG"
    assert await redis.get("tax:UniRef50_UPI0001") == b"9606"
    assert await redis.get("tax:UniRef50_UPI0002") is None


async def test_build_respects_custom_key_format(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A TaxID=42\nMKT\n")
    redis = FakeAsyncRedis()

    await build(
        records=iter_fasta(fasta),
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
        records=iter_fasta(fasta),
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
        await build(records=iter_fasta(fasta))


async def test_build_from_fasta_shim(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">A TaxID=7227\nMKT\n")
    redis = FakeAsyncRedis()

    seq_count, tax_count = await build_from_fasta(fasta_path=fasta, redis=redis)
    assert (seq_count, tax_count) == (1, 1)
    assert await redis.get("seq:A") == b"MKT"
