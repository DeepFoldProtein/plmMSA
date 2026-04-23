from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(slots=True)
class AlignmentHit:
    """One pairwise alignment hit to fold into an A3M.

    `columns` is the aligner output: list of `(query_idx, target_idx)` pairs
    where `-1` marks a gap on that side. The list is ordered from start to
    end of the alignment.
    """

    target_id: str
    score: float
    target_seq: str
    columns: Sequence[tuple[int, int]]


def render_hit(query_len: int, hit: AlignmentHit) -> str:
    """Render one hit as an A3M row.

    Rules we follow (matches upstream A3M convention):

    - The row has `query_len` uppercase slots — one per query column. Every
      slot is a target residue (uppercase) if that column is a match, or `-`
      if the column is a gap in the target or falls outside the alignment.
    - Residues aligned against nothing in the query (gaps in query) are
      emitted as lowercase *between* slots.
    """
    slots: list[str] = ["-"] * query_len
    inserts_before: list[list[str]] = [[] for _ in range(query_len + 1)]
    last_qi = -1

    for qi, ti in hit.columns:
        if qi >= 0 and ti >= 0:
            if 0 <= qi < query_len:
                slots[qi] = hit.target_seq[ti].upper()
            last_qi = qi
        elif qi >= 0 and ti < 0:
            last_qi = qi
        elif qi < 0 and ti >= 0:
            pos = last_qi + 1
            pos = max(0, min(pos, query_len))
            inserts_before[pos].append(hit.target_seq[ti].lower())

    pieces: list[str] = []
    pieces.extend(inserts_before[0])
    for qi in range(query_len):
        pieces.append(slots[qi])
        pieces.extend(inserts_before[qi + 1])
    return "".join(pieces)


def assemble_a3m(
    *,
    query_id: str,
    query_seq: str,
    query_self_score: float,
    hits: Sequence[AlignmentHit],
) -> str:
    """Assemble a full A3M from one query + a ranked list of hits.

    First record is the query itself (ungapped, scored); each hit renders as
    a same-length uppercase row plus lowercase inserts (see `render_hit`).
    """
    lines: list[str] = []
    lines.append(f">{query_id}   {query_self_score:.3f}")
    lines.append(query_seq)
    for hit in hits:
        lines.append(f">{hit.target_id}   {hit.score:.3f}")
        lines.append(render_hit(len(query_seq), hit))
    return "\n".join(lines) + "\n"


def assemble_paired_a3m(
    *,
    query_ids: Sequence[str],
    query_seqs: Sequence[str],
    paired_rows: Sequence[tuple[str, Sequence[AlignmentHit], float]],
    query_self_score: float | None = None,
) -> str:
    """Assemble a paired A3M from per-chain queries + ranked paired rows.

    `paired_rows[i]` is `(taxonomy_id, chain_hits, joint_score)`, where
    `chain_hits[c]` is the chosen hit for chain `c` (same order as
    `query_ids` / `query_seqs`). Each paired row emits one concatenated
    A3M record: per-chain aligned slots separated by a gap run of length
    `max(chain_lengths) // 10` (ColabFold's paired-A3M convention), so
    downstream tools that split paired rows back into chains know where
    the boundaries live.

    The first record is the concatenated query with the same separator.
    """
    if len(query_ids) != len(query_seqs):
        raise ValueError(f"paired A3M: {len(query_ids)} query_ids vs {len(query_seqs)} query_seqs")
    if not query_seqs:
        return ""

    sep_len = max(len(s) for s in query_seqs) // 10
    sep = "-" * sep_len

    lines: list[str] = []
    # Query row — id is `|`-joined so clients can recover per-chain ids.
    joined_query_id = "|".join(query_ids)
    # Defaults to the legacy sum-of-lengths convention for direct callers;
    # orchestrator jobs pass aligner-evaluated self scores.
    self_score = (
        float(sum(len(s) for s in query_seqs))
        if query_self_score is None
        else float(query_self_score)
    )
    lines.append(f">{joined_query_id}   {self_score:.3f}")
    lines.append(sep.join(query_seqs))

    for tax_id, chain_hits, joint_score in paired_rows:
        if len(chain_hits) != len(query_seqs):
            raise ValueError(
                f"paired A3M: row has {len(chain_hits)} hits, expected {len(query_seqs)}"
            )
        rendered = sep.join(
            render_hit(len(qs), hit) for qs, hit in zip(query_seqs, chain_hits, strict=True)
        )
        # Header: taxonomy id + per-chain target ids (|-joined) + joint score.
        chain_labels = "|".join(h.target_id for h in chain_hits)
        lines.append(f">tax:{tax_id}|{chain_labels}   {joint_score:.3f}")
        lines.append(rendered)
    return "\n".join(lines) + "\n"
