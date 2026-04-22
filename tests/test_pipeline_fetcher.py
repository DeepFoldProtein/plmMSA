from __future__ import annotations

from pathlib import Path

from plmmsa.pipeline.fetcher import (
    DictTargetFetcher,
    FastaTargetFetcher,
    dict_fetcher_from_fasta_text,
)


async def test_dict_fetcher_drops_unknown_ids() -> None:
    fetcher = DictTargetFetcher({"A": "MKT", "B": "AAA"})
    out = await fetcher.fetch("ankh_uniref50", ["A", "UNKNOWN", "B"])
    assert out == {"A": "MKT", "B": "AAA"}


async def test_fasta_fetcher_roundtrip(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">X\nMKT\n>Y  description ignored\nAAA\nGGG\n")
    fetcher = FastaTargetFetcher(fasta)
    out = await fetcher.fetch("c", ["X", "Y", "Z"])
    assert out == {"X": "MKT", "Y": "AAAGGG"}


async def test_dict_fetcher_from_fasta_text() -> None:
    text = ">A\nMKT\n>B\nAAA\n"
    fetcher = dict_fetcher_from_fasta_text(text)
    out = await fetcher.fetch("c", ["A", "B"])
    assert out == {"A": "MKT", "B": "AAA"}
