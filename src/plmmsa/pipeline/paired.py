"""Paired-MSA taxonomy join.

Translates MMseqs2's `PairAlign` contract onto our per-chain hit lists:

1. Each chain contributes a flat list of `AlignmentHit` records with a
   taxonomy id resolved from the UniRef50 cluster representative
   (`tax:UniRef50_<acc>` in `cache-seq`).
2. Bucket per-chain hits by taxonomy. Drop hits with no tax record.
3. For each taxonomy that appears in *every* chain, pick the highest-
   scoring hit per chain — that yields one paired row per shared
   taxonomy. Taxonomies missing from any chain are dropped.
4. Rank paired rows by joint score (sum of per-chain hit scores).

Nothing here knows about PLMs, alignment internals, or A3M rendering —
the caller passes `AlignmentHit`s and a parallel `tax_by_id` mapping,
and gets back a ranked `list[PairedRow]` plus bookkeeping counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from plmmsa.pipeline.a3m import AlignmentHit


@dataclass(slots=True, frozen=True)
class PairedRow:
    """One paired row across all chains, keyed by a shared taxonomy.

    `hits[c]` is the chosen hit for chain `c`; `joint_score` is the sum of
    `hit.score` across chains (what we rank by when sorting the paired
    A3M for the client).
    """

    taxonomy_id: str
    hits: tuple[AlignmentHit, ...]
    joint_score: float


@dataclass(slots=True, frozen=True)
class PairedJoinResult:
    """What the join produced, plus counters callers surface in stats."""

    rows: list[PairedRow]
    # Per-chain (n_hits_in, n_hits_with_tax) so stats can show how much
    # of each chain's pool made it into the pairing-eligible bucket.
    per_chain_in: tuple[int, ...]
    per_chain_with_tax: tuple[int, ...]
    # Taxonomy set sizes at each stage — useful for understanding why a
    # join produced zero rows.
    taxonomies_per_chain: tuple[int, ...]
    shared_taxonomies: int


def join_by_taxonomy(
    per_chain_hits: Sequence[Sequence[AlignmentHit]],
    per_chain_tax: Sequence[dict[str, str]],
) -> PairedJoinResult:
    """Join per-chain hit lists on shared taxonomy ids.

    `per_chain_hits[c]` is chain c's hit list; `per_chain_tax[c]` maps
    `target_id → taxonomy_id`. An id missing from the tax map is dropped
    before bucketing.
    """
    if len(per_chain_hits) != len(per_chain_tax):
        raise ValueError(
            f"paired join: per_chain_hits has {len(per_chain_hits)} entries "
            f"but per_chain_tax has {len(per_chain_tax)}"
        )
    if not per_chain_hits:
        return PairedJoinResult(
            rows=[],
            per_chain_in=(),
            per_chain_with_tax=(),
            taxonomies_per_chain=(),
            shared_taxonomies=0,
        )

    # Step 1 — per-chain: bucket hits by taxonomy, keep only the
    # highest-scoring hit per (chain, taxonomy).
    per_chain_best: list[dict[str, AlignmentHit]] = []
    per_chain_in: list[int] = []
    per_chain_with_tax: list[int] = []
    for hits, tax_map in zip(per_chain_hits, per_chain_tax, strict=True):
        per_chain_in.append(len(hits))
        best: dict[str, AlignmentHit] = {}
        with_tax = 0
        for h in hits:
            tax_id = tax_map.get(h.target_id)
            if not tax_id:
                continue
            with_tax += 1
            existing = best.get(tax_id)
            if existing is None or h.score > existing.score:
                best[tax_id] = h
        per_chain_best.append(best)
        per_chain_with_tax.append(with_tax)

    taxonomies_per_chain = tuple(len(b) for b in per_chain_best)

    # Step 2 — intersect taxonomies across all chains. Python's set.
    # intersection_update handles any arity.
    if any(not b for b in per_chain_best):
        shared: set[str] = set()
    else:
        shared = set(per_chain_best[0].keys())
        for other in per_chain_best[1:]:
            shared &= other.keys()

    # Step 3 — emit a PairedRow per shared taxonomy, ranked by joint score.
    rows: list[PairedRow] = []
    for tax_id in shared:
        chain_hits = tuple(b[tax_id] for b in per_chain_best)
        joint = sum(h.score for h in chain_hits)
        rows.append(PairedRow(taxonomy_id=tax_id, hits=chain_hits, joint_score=joint))
    rows.sort(key=lambda r: r.joint_score, reverse=True)

    return PairedJoinResult(
        rows=rows,
        per_chain_in=tuple(per_chain_in),
        per_chain_with_tax=tuple(per_chain_with_tax),
        taxonomies_per_chain=taxonomies_per_chain,
        shared_taxonomies=len(shared),
    )


__all__ = ["PairedJoinResult", "PairedRow", "join_by_taxonomy"]
