"""CI-side sanity check for the regression fixtures.

The numeric regression (did our stack's MSA for T1104 stay close to the
recorded baseline?) requires a live pipeline and runs out-of-band via
`bench/run_regression.py`. This test just verifies that the in-tree fixtures
are structurally intact so the bench script can't be silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "casp15"
_REQUIRED_STAT_KEYS = {
    "query_len",
    "msa_depth",
    "scored_hits",
    "top_hit",
    "second_hit",
    "tolerance",
}
_REQUIRED_HIT_KEYS = {"id", "score"}
_REQUIRED_TOLERANCE_KEYS = {"depth_pct", "score_abs"}


def _load_expected() -> dict:
    return json.loads((_FIXTURE_DIR / "expected_stats.json").read_text())


def test_fixture_dir_exists() -> None:
    assert _FIXTURE_DIR.is_dir()
    assert (_FIXTURE_DIR / "expected_stats.json").is_file()


def test_expected_stats_shape() -> None:
    data = _load_expected()
    targets = data["targets"]
    assert targets, "at least one target expected"
    for name, stats in targets.items():
        missing = _REQUIRED_STAT_KEYS - stats.keys()
        assert not missing, f"{name}: missing keys {missing}"
        for hit_key in ("top_hit", "second_hit"):
            assert stats[hit_key].keys() >= _REQUIRED_HIT_KEYS, f"{name}.{hit_key}"
        assert stats["tolerance"].keys() >= _REQUIRED_TOLERANCE_KEYS


@pytest.mark.parametrize("target", sorted(_load_expected()["targets"].keys()))
def test_fasta_matches_query_len(target: str) -> None:
    data = _load_expected()
    expected_len = data["targets"][target]["query_len"]

    fasta = (_FIXTURE_DIR / f"{target}.fasta").read_text()
    lines = [ln.strip() for ln in fasta.splitlines() if ln.strip()]
    assert lines[0].startswith(">"), f"{target}.fasta has no FASTA header"
    seq = "".join(lines[1:])
    assert len(seq) == expected_len, (
        f"{target}: expected_stats.query_len={expected_len} but fasta has {len(seq)} residues"
    )
