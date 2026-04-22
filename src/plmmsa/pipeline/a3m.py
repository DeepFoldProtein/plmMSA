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
