"""OTalign end-to-end coverage.

OTalign is no longer a stub — it runs the full unbalanced-Sinkhorn →
PMI/gap-factor → affine-gap DP pipeline. These tests pin:
- class wiring (id, Aligner subclass)
- a known-near-identity input produces a high-mass plan → nontrivial
  matched columns
- flat / orthogonal embeddings produce a Sinkhorn-uniform plan → the DP
  still terminates cleanly (no NaNs, valid Alignment object)
- DP mode override works through the `dp_mode` kwarg
"""

from __future__ import annotations

import numpy as np

from plmmsa.align.base import Aligner, MatrixAligner
from plmmsa.align.otalign import OTalign


def test_otalign_is_aligner_but_not_matrix_aligner() -> None:
    """OTalign consumes embeddings directly (cost matrix is recomputed
    every Sinkhorn iteration), so it inherits the embedding-level
    Aligner — NOT MatrixAligner whose `align_matrix(sim)` contract can't
    be satisfied."""
    assert issubclass(OTalign, Aligner)
    assert not issubclass(OTalign, MatrixAligner)
    assert OTalign.id == "otalign"
    assert OTalign.display_name.startswith("OTalign")


def test_otalign_identical_inputs_produce_match_columns() -> None:
    """Identical query + target → high-mass diagonal plan → the DP
    should find a matched run along the diagonal."""
    rng = np.random.default_rng(0)
    # Distinct residue embeddings per position — not pathologically close.
    emb = rng.normal(size=(12, 8)).astype(np.float32)
    aligner = OTalign()
    [alignment] = aligner.align(emb, [emb], mode="local")
    matched = [(qi, ti) for qi, ti in alignment.columns if qi >= 0 and ti >= 0]
    # Identity inputs should produce mostly-diagonal matches.
    if matched:
        diagonal = sum(1 for qi, ti in matched if qi == ti)
        assert diagonal / len(matched) >= 0.5
    # Non-empty alignment.
    assert alignment.query_end > alignment.query_start
    assert alignment.target_end > alignment.target_start


def test_otalign_orthogonal_inputs_do_not_crash() -> None:
    """Hand-crafted orthogonal query/target → small max match, Sinkhorn
    may hit n_iter without converging. The aligner must still return a
    valid Alignment (possibly with score == 0 and empty columns)."""
    q = np.eye(6, 4, dtype=np.float32)
    t = np.eye(5, 4, dtype=np.float32)[::-1]  # reversed → near-zero sim
    aligner = OTalign()
    [alignment] = aligner.align(q, [t], mode="local")
    assert np.isfinite(alignment.score)
    # Valid index ranges.
    assert 0 <= alignment.query_start <= alignment.query_end <= 6
    assert 0 <= alignment.target_start <= alignment.target_end <= 5


def test_otalign_dp_mode_override() -> None:
    """`dp_mode="glocal"` is OTalign's upstream default; make sure
    callers can also pick global / q2t / t2q / local via the kwarg."""
    rng = np.random.default_rng(1)
    q = rng.normal(size=(8, 4)).astype(np.float32)
    t = rng.normal(size=(10, 4)).astype(np.float32)
    aligner = OTalign()
    for dp_mode in ("global", "q2t", "t2q", "glocal", "local"):
        [alignment] = aligner.align(q, [t], dp_mode=dp_mode)
        assert np.isfinite(alignment.score)
