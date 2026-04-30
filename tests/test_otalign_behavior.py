"""OTalign behavior pins for the templates_realign use case.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.1 — pin OTalign's actual behavior
on synthetic embeddings so the templates re-align pipeline (which is built
on top) has a stable contract to test against.

Three families of pins:

1. **Identity / substring scenarios** — when the template is a contiguous
   slice of the query, OTalign should recover that slice with high
   accuracy across modes; deviations show up as test failures rather than
   silent quality drops downstream.

2. **Mode contract** — what each `AlignMode` actually guarantees about
   the trace's start / end indices. The doc-comment in `align/base.py`
   uses "free query end-gap" wording that does NOT match the
   implementation; we pin what the *code* does, with comments noting the
   discrepancy. The pipeline keys on these guarantees (e.g. q2t pinning
   `query_end == Lq`).

3. **Robustness** — random / orthogonal templates, degenerate sizes, bound
   safety, determinism. These don't assert biology — they assert that the
   aligner returns a structurally valid `Alignment` for any input.

All tests use small embedding dim (D=32) and short sequences (Lq ≤ 80) so
this file finishes in a few seconds. We're testing structure, not
biology — see `tests/test_otalign_real_embeddings.py` (RUN_SLOW=1) for
the Ankh-Large pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align.base import AlignMode
from plmmsa.align.otalign import OTalign

D = 32  # embedding dim — small enough to keep tests fast


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _columns_in_bounds(columns, lq: int, lt: int) -> bool:
    """Every (qi, ti) is inside the input shape and not both -1."""
    for qi, ti in columns:
        if qi < -1 or qi >= lq:
            return False
        if ti < -1 or ti >= lt:
            return False
        if qi == -1 and ti == -1:
            return False
    return True


def _matched(columns) -> list[tuple[int, int]]:
    return [(qi, ti) for qi, ti in columns if qi >= 0 and ti >= 0]


# ---------------------------------------------------------------------------
# 1) Identity & substring scenarios
# ---------------------------------------------------------------------------


def test_identical_inputs_q2t_is_diagonal() -> None:
    """`q == t` under q2t collapses to the identity permutation.

    This is the strongest guarantee any aligner gives — same residues on
    both sides, every position matches itself. If this drifts, all
    downstream identity tests are unreliable.
    """
    rng = _rng(0)
    Lq = 64
    q = rng.normal(size=(Lq, D)).astype(np.float32)

    [a] = OTalign().align(q, [q], mode="q2t")

    matched = _matched(a.columns)
    assert len(matched) == Lq, "every position must match"
    assert all(qi == ti for qi, ti in matched), "matches must lie on the diagonal"
    assert a.query_start == 0 and a.query_end == Lq
    assert a.target_start == 0 and a.target_end == Lq
    assert a.score > 0


def test_substring_glocal_recovers_offset_diagonal() -> None:
    """`t = q[a:b]` under glocal recovers the strict (a, 0) diagonal.

    glocal allows free end-gaps on both sides, so the alignment trims to
    exactly the substring without the forced-query-end artifact that q2t
    introduces (see `test_substring_q2t_pulls_last_column_to_query_end`).
    """
    rng = _rng(1)
    Lq, a, b = 80, 20, 60
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = q[a:b].copy()

    [aln] = OTalign().align(q, [t], mode="glocal")

    matched = _matched(aln.columns)
    assert len(matched) == (b - a), "every template residue should match"
    assert all(qi - ti == a for qi, ti in matched)
    assert aln.query_start == a and aln.query_end == b
    assert aln.target_start == 0 and aln.target_end == (b - a)


def test_substring_q2t_pulls_last_column_to_query_end() -> None:
    """`t = q[a:b]` under q2t — forced `query_end == Lq` pulls the last
    template residue to the last query column.

    This is a real artifact of q2t the templates_realign pipeline must
    expect: in the rendered A3M row the last template residue can land
    in a far-right slot rather than its substring-matched position.
    Most (≥ 95%) matches still lie on the (a, 0) offset — the artifact
    affects only the trailing residue or two.
    """
    rng = _rng(1)
    Lq, a, b = 80, 20, 60
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = q[a:b].copy()

    [aln] = OTalign().align(q, [t], mode="q2t")

    matched = _matched(aln.columns)
    assert len(matched) == (b - a), "every template residue placed (template-global)"
    assert aln.target_start == 0 and aln.target_end == (b - a)
    assert aln.query_end == Lq, "q2t must consume query to its right end"
    on_offset = sum(1 for qi, ti in matched if qi - ti == a)
    assert on_offset / len(matched) >= 0.95


# ---------------------------------------------------------------------------
# 2) Mode contract — what does each mode actually guarantee?
# ---------------------------------------------------------------------------
#
# NOTE: `align/base.py` describes q2t as "free query end-gap only (template
# is global)". The implementation in `align/otalign_dp.py` does the
# opposite: q2t pins query_end == Lq (last-row traceback start) and frees
# query_start (M[i, 0] = 0). It is pinned here as such — fix the
# docstring or the code, but the templates_realign pipeline reads the
# implementation, so that is what we lock down.


def _build_substring(seed: int):
    rng = _rng(seed)
    Lq, a, b = 80, 20, 60
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = q[a:b].copy()
    return q, t, Lq, a, b


@pytest.mark.parametrize("mode", ["local", "global", "glocal", "q2t", "t2q"])
def test_columns_always_in_bounds(mode: AlignMode) -> None:
    """Universal invariant — no matter the mode, the column list is a
    well-formed alignment path: indices in range, never both -1."""
    q, t, Lq, _, _ = _build_substring(seed=10)
    [aln] = OTalign().align(q, [t], mode=mode)
    assert _columns_in_bounds(aln.columns, Lq, t.shape[0])
    # Index bookkeeping — start/end derive from columns, so they must be
    # consistent.
    assert 0 <= aln.query_start <= aln.query_end <= Lq
    assert 0 <= aln.target_start <= aln.target_end <= t.shape[0]
    assert np.isfinite(aln.score)


def test_q2t_pins_query_end_to_Lq() -> None:
    """q2t traceback starts at argmax over the last row → query_end == Lq.

    This is the contract the templates_realign renderer assumes: the
    template's last column (or beyond) lands somewhere in the query, and
    the query is consumed all the way to its right end.
    """
    q, t, Lq, _, _ = _build_substring(seed=11)
    [aln] = OTalign().align(q, [t], mode="q2t")
    assert aln.query_end == Lq


def test_global_consumes_both_sides() -> None:
    """global pins both ends on both sides (NW). Used in tests that need
    the strongest end-gap policy (e.g. self-score baselines)."""
    q, t, Lq, _, _ = _build_substring(seed=12)
    [aln] = OTalign().align(q, [t], mode="global")
    assert aln.query_start == 0 and aln.query_end == Lq
    assert aln.target_start == 0 and aln.target_end == t.shape[0]


def test_t2q_pins_target_end_to_Lt() -> None:
    """t2q is the mirror of q2t — pins target_end at Lt."""
    q, t, _, _, _ = _build_substring(seed=13)
    [aln] = OTalign().align(q, [t], mode="t2q")
    assert aln.target_end == t.shape[0]


# ---------------------------------------------------------------------------
# 3) Robustness — random templates, degenerate sizes, determinism
# ---------------------------------------------------------------------------


def test_orthogonal_template_does_not_crash() -> None:
    """A template drawn from a fresh RNG has near-zero similarity with
    the query. OTalign must still return a structurally-valid Alignment
    — possibly empty, possibly tiny, but always finite + bounded."""
    rng = _rng(20)
    Lq, Lt = 80, 40
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = rng.normal(size=(Lt, D)).astype(np.float32)

    [aln] = OTalign().align(q, [t], mode="q2t")

    assert np.isfinite(aln.score)
    assert _columns_in_bounds(aln.columns, Lq, Lt)


def test_score_separates_signal_from_noise() -> None:
    """Identity score (q vs q) must be strictly higher than noise score
    (q vs random). This is what the OTalign filter_threshold relies on
    to discard low-quality hits.
    """
    rng = _rng(21)
    Lq, Lt = 64, 64
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    noise = rng.normal(size=(Lt, D)).astype(np.float32)

    aligner = OTalign()
    [signal] = aligner.align(q, [q], mode="q2t")
    [bg] = aligner.align(q, [noise], mode="q2t")
    assert signal.score > bg.score


def test_determinism_same_call_same_result() -> None:
    """Two identical calls produce byte-identical alignments. Caching
    relies on this — if OTalign is non-deterministic, the same query +
    template hashed cache entry would disagree across regenerations."""
    rng = _rng(30)
    Lq = 32
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = q[5:25].copy()

    aligner = OTalign()
    [a1] = aligner.align(q, [t], mode="q2t")
    [a2] = aligner.align(q, [t], mode="q2t")
    assert a1.columns == a2.columns
    assert a1.score == pytest.approx(a2.score, abs=0, rel=0)
    assert (a1.query_start, a1.query_end) == (a2.query_start, a2.query_end)
    assert (a1.target_start, a1.target_end) == (a2.target_start, a2.target_end)


@pytest.mark.parametrize(
    "Lq,Lt",
    [
        (1, 1),  # both degenerate
        (1, 10),  # query is a single residue
        (10, 1),  # template is a single residue
        (1022, 5),  # max query length supported by Ankh-Large
    ],
)
def test_degenerate_sizes(Lq: int, Lt: int) -> None:
    """Edge sizes — single-residue inputs and the Ankh-Large max length.
    Must not crash, must return bounded columns and a finite score."""
    rng = _rng(40)
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t = rng.normal(size=(Lt, D)).astype(np.float32)

    [aln] = OTalign().align(q, [t], mode="q2t")

    assert np.isfinite(aln.score)
    assert _columns_in_bounds(aln.columns, Lq, Lt)


def test_target_indices_match_target_start_end_attributes() -> None:
    """`target_start` / `target_end` reported on the Alignment must
    bracket exactly the non-gap target indices in `columns` (DPResult
    derives them from the columns themselves; pin so a refactor that
    starts caching them separately doesn't drift)."""
    q, t, _, _, _ = _build_substring(seed=50)
    [aln] = OTalign().align(q, [t], mode="q2t")

    t_idxs = sorted({ti for _, ti in aln.columns if ti >= 0})
    assert t_idxs[0] == aln.target_start
    assert t_idxs[-1] + 1 == aln.target_end

    q_idxs = sorted({qi for qi, _ in aln.columns if qi >= 0})
    assert q_idxs[0] == aln.query_start
    assert q_idxs[-1] + 1 == aln.query_end


def test_batch_against_multiple_targets_is_independent() -> None:
    """Aligning N targets in one call must produce the same per-target
    results as aligning each target individually. The pipeline batches
    by default; if batching introduced cross-target leakage (e.g.
    shared in-place state in the Sinkhorn solver) it would surface here."""
    rng = _rng(60)
    Lq = 64
    q = rng.normal(size=(Lq, D)).astype(np.float32)
    t1 = q[10:40].copy()
    t2 = q[20:50].copy()
    t3 = rng.normal(size=(25, D)).astype(np.float32)

    aligner = OTalign()
    batch = aligner.align(q, [t1, t2, t3], mode="q2t")
    [s1] = aligner.align(q, [t1], mode="q2t")
    [s2] = aligner.align(q, [t2], mode="q2t")
    [s3] = aligner.align(q, [t3], mode="q2t")

    for from_batch, from_solo in zip(batch, [s1, s2, s3]):
        assert from_batch.columns == from_solo.columns
        assert from_batch.score == pytest.approx(from_solo.score, abs=0, rel=0)
