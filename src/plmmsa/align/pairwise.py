"""Pairwise alignment orchestrator.

Takes a query sequence + one or more target sequences (FASTA-shape input,
not raw embeddings) and runs OTalign / PLMAlign over PLM embeddings,
returning per-target alignment columns + score + spans. Optionally
renders the result as a ColabFold/AlphaFold-style A3M: the query at
index 0, followed by one row per kept target with insertions kept as
lowercase **between** the first and last matched query column
(leading/trailing inserts trimmed).

Shares the embed-then-align skeleton with `plmmsa.templates.pipeline`
but skips the hmmsearch A3M parsing — input here is already
structured — and uses its own row renderer (templates/realign drops
all lowercase to match hmmsearch shape; this endpoint preserves
middle lowercase since the typical consumer is ColabFold/AlphaFold).
The `TemplatesTransport` interface is reused since the wire calls
(`embed` + `align`) are identical.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.templates.transport import TemplatesTransport

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PairwiseAlignTarget:
    target_id: str
    sequence: str


@dataclass(slots=True)
class PairwiseAlignConfig:
    """Operator-tunable knobs. `default_*` mirror the API edge defaults
    — callers can override per request. `max_query_length` and
    `max_records` are hard ceilings producing typed errors when
    exceeded.
    """

    default_aligner: str = "otalign"
    default_model: str = "ankh_large"
    default_mode: str = "glocal"
    max_query_length: int = 1022  # min across PLMs (ESM1b's positional cap)
    max_records: int = 5000


@dataclass(slots=True)
class PairwiseAlignRequest:
    query_id: str
    query_sequence: str
    targets: list[PairwiseAlignTarget]
    model: str | None = None
    mode: str | None = None
    aligner: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    # When True, render each kept alignment to an A3M row (target residues
    # placed on query columns) and assemble the rows into `result.a3m`.
    # `columns` is always returned regardless — A3M is a derived view.
    emit_a3m: bool = True
    # When True, results are emitted in OTalign-score-descending order
    # (best hit first). When False (default), input order is preserved.
    sort_by_score: bool = False


@dataclass(slots=True, frozen=True)
class PairwiseAlignment:
    target_id: str
    score: float
    # Each `(qi, ti)` is a column in the OTalign path. `-1` on either
    # side marks a gap on that side. Indices are 0-based into the
    # normalized query / target sequences.
    columns: list[tuple[int, int]]
    # Half-open spans `[start, end)` over the matched columns.
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    # Rendered only when `emit_a3m=True` was set on the request.
    # Shape: one uppercase residue or `-` per query column, plus
    # lowercase insertions kept *between* the first and last matched
    # query column. Insertions outside the matched span are trimmed
    # — see `_render_row_middle_inserts` for the exact rule.
    # `len(a3m_row) >= query_length`; the difference equals the count
    # of kept lowercase insertions for this row.
    a3m_row: str | None


@dataclass(slots=True)
class PairwiseAlignResult:
    alignments: list[PairwiseAlignment]
    stats: dict[str, Any]
    # None when emit_a3m=False. Otherwise contains at minimum the
    # `>{query_id}\n{query_sequence}\n` record — even when every target
    # was dropped as no-match.
    a3m: str | None


class PairwiseAlignOrchestrator:
    """Single-shot pairwise-align orchestrator. Construct once per
    process; one instance handles many requests sequentially or
    concurrently (it carries no per-request state).
    """

    def __init__(
        self,
        *,
        config: PairwiseAlignConfig,
        transport: TemplatesTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    async def run(self, request: PairwiseAlignRequest) -> PairwiseAlignResult:
        cfg = self._config

        # Step 1 — normalize query and check shape / limits.
        query_seq = _normalize(request.query_sequence)
        if not query_seq:
            raise PlmMSAError(
                "query_sequence is empty after normalization.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
            )
        if not _is_uppercase_alpha(query_seq):
            raise PlmMSAError(
                "query_sequence contains non-amino-acid characters.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
            )
        if len(query_seq) > cfg.max_query_length:
            raise PlmMSAError(
                f"query_sequence length {len(query_seq)} exceeds max {cfg.max_query_length}.",
                code=ErrorCode.SEQ_TOO_LONG,
                http_status=400,
                detail={"length": len(query_seq), "max": cfg.max_query_length},
            )

        if not request.targets:
            raise PlmMSAError(
                "targets is empty.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
            )
        if len(request.targets) > cfg.max_records:
            raise PlmMSAError(
                f"targets has {len(request.targets)} entries; max is {cfg.max_records}.",
                code=ErrorCode.QUEUE_FULL,
                http_status=413,
                detail={"records": len(request.targets), "max": cfg.max_records},
            )

        # Step 2 — normalize + validate each target. Reject duplicate ids
        # up front: the response is keyed by `target_id`, so duplicates
        # would silently produce ambiguous output.
        targets_normalized: list[tuple[str, str]] = []  # (id, raw seq)
        seen_ids: set[str] = set()
        for t in request.targets:
            tid = t.target_id
            if tid in seen_ids:
                raise PlmMSAError(
                    f"duplicate target_id '{tid}'.",
                    code=ErrorCode.INVALID_FASTA,
                    http_status=400,
                    detail={"target_id": tid},
                )
            seen_ids.add(tid)

            raw = _normalize(t.sequence)
            if not raw:
                raise PlmMSAError(
                    f"target '{tid}' sequence is empty after normalization.",
                    code=ErrorCode.INVALID_FASTA,
                    http_status=400,
                    detail={"target_id": tid},
                )
            if not _is_uppercase_alpha(raw):
                raise PlmMSAError(
                    f"target '{tid}' contains non-amino-acid characters.",
                    code=ErrorCode.INVALID_FASTA,
                    http_status=400,
                    detail={"target_id": tid},
                )
            if len(raw) > cfg.max_query_length:
                raise PlmMSAError(
                    f"target '{tid}' length {len(raw)} exceeds max {cfg.max_query_length}.",
                    code=ErrorCode.SEQ_TOO_LONG,
                    http_status=400,
                    detail={
                        "target_id": tid,
                        "length": len(raw),
                        "max": cfg.max_query_length,
                    },
                )
            targets_normalized.append((tid, raw))

        model = request.model or cfg.default_model
        mode = request.mode or cfg.default_mode
        aligner = request.aligner or cfg.default_aligner

        # Step 3 — dedup target residues. Many fan-out workloads embed
        # the same canonical chain under different ids; embedding the
        # unique set is the obvious win.
        seq_to_idx: dict[str, int] = {}
        unique_seqs: list[str] = []
        for _, raw in targets_normalized:
            if raw not in seq_to_idx:
                seq_to_idx[raw] = len(unique_seqs)
                unique_seqs.append(raw)

        # Step 4 — embed query + unique targets in one call. Query lands
        # at index 0 of the result tensor list.
        all_seqs = [query_seq, *unique_seqs]
        embeddings = await self._transport.embed(model=model, sequences=all_seqs)
        if len(embeddings) != len(all_seqs):
            raise PlmMSAError(
                "embedding service returned a different number of tensors than requested.",
                code=ErrorCode.INTERNAL,
                http_status=502,
                detail={"sent": len(all_seqs), "got": len(embeddings)},
            )
        query_emb = embeddings[0]
        unique_embs = embeddings[1:]
        if query_emb.shape[0] != len(query_seq):
            raise PlmMSAError(
                "query embedding length does not match query sequence.",
                code=ErrorCode.INTERNAL,
                http_status=502,
                detail={
                    "query_length": len(query_seq),
                    "embedding_length": int(query_emb.shape[0]),
                },
            )
        for seq, emb in zip(unique_seqs, unique_embs, strict=True):
            if emb.shape[0] != len(seq):
                raise PlmMSAError(
                    "target embedding length does not match its sequence.",
                    code=ErrorCode.INTERNAL,
                    http_status=502,
                    detail={"expected": len(seq), "got": int(emb.shape[0])},
                )

        # Step 5 — fan unique embeddings back out to the per-target list
        # (preserves input order).
        target_embs = [unique_embs[seq_to_idx[raw]] for _, raw in targets_normalized]

        # Step 6 — one batched align call.
        alignments_raw = await self._transport.align(
            aligner=aligner,
            mode=mode,
            query_embedding=query_emb,
            target_embeddings=target_embs,
            options=dict(request.options),
        )
        if len(alignments_raw) != len(targets_normalized):
            raise PlmMSAError(
                "align service returned a different number of alignments than requested.",
                code=ErrorCode.INTERNAL,
                http_status=502,
                detail={
                    "sent": len(targets_normalized),
                    "got": len(alignments_raw),
                },
            )

        # Step 7 — assemble per-target results. Records where the
        # aligner produced zero match columns are dropped from the
        # output but counted in stats.
        results: list[PairwiseAlignment] = []
        targets_dropped_no_match = 0
        query_len = len(query_seq)
        for (tid, raw), aln in zip(targets_normalized, alignments_raw, strict=True):
            cols = [(int(c[0]), int(c[1])) for c in aln.get("columns") or []]
            has_match = any(qi >= 0 and ti >= 0 for qi, ti in cols)
            if not has_match:
                targets_dropped_no_match += 1
                continue
            score = float(aln.get("score", 0.0))
            row = _render_row_middle_inserts(query_len, raw, cols) if request.emit_a3m else None
            results.append(
                PairwiseAlignment(
                    target_id=tid,
                    score=score,
                    columns=cols,
                    query_start=int(aln.get("query_start", 0)),
                    query_end=int(aln.get("query_end", 0)),
                    target_start=int(aln.get("target_start", 0)),
                    target_end=int(aln.get("target_end", 0)),
                    a3m_row=row,
                )
            )

        # Optional re-order. Stable sort so equal scores keep their
        # input-order tiebreak.
        if request.sort_by_score:
            results.sort(key=lambda r: -r.score)

        # Step 8 — assemble the optional A3M payload. ColabFold/AlphaFold
        # A3M shape: the query is the first record, followed by one
        # `>id` + row pair per kept alignment. Score stamped at three
        # decimals to match the templates/realign convention.
        a3m: str | None = None
        if request.emit_a3m:
            lines: list[str] = [f">{request.query_id}", query_seq]
            for r in results:
                if r.a3m_row is None:
                    continue
                lines.append(f">{r.target_id} score:{r.score:.3f}")
                lines.append(r.a3m_row)
            a3m = "\n".join(lines) + "\n"

        stats = {
            "pipeline": "align_pairwise",
            "query_id": request.query_id,
            "query_length": query_len,
            "targets_in": len(targets_normalized),
            "targets_kept": len(results),
            "targets_dropped_no_match": targets_dropped_no_match,
            "unique_target_seqs": len(unique_seqs),
            "model": model,
            "mode": mode,
            "aligner": aligner,
            "sort_by_score": request.sort_by_score,
            "emit_a3m": request.emit_a3m,
        }
        return PairwiseAlignResult(alignments=results, stats=stats, a3m=a3m)


def _render_row_middle_inserts(
    query_len: int,
    target_seq: str,
    columns: Sequence[tuple[int, int]],
) -> str:
    """Render an A3M row with insertions kept only between matches.

    Columns are OTalign's alignment path: each `(qi, ti)` is one
    column. `-1` marks a gap on that side. Three column kinds are
    handled:

      - `qi>=0, ti>=0` (match): place `target_seq[ti].upper()` at slot
        `qi`. These are the match-state columns.
      - `qi>=0, ti=-1` (target gap on query column): leave slot `qi`
        as `-`. Advances the "last matched query column" cursor.
      - `qi=-1, ti>=0` (insertion in target relative to query): collect
        as a lowercase residue at the cursor's current insert position.

    Insert handling is the key policy: insertions are kept **only
    between** the first and last matched query column. Leading
    insertions (those that land before the first match) and trailing
    insertions (after the last match) are dropped. The motivation: the
    aligner often places noisy fragments on either side of a coherent
    match block; those are not informative for downstream MSA
    consumers and pad rows uselessly.

    Output: `query_len` aligned slots (uppercase residues or `-`) plus
    interleaved lowercase insert runs. `len(output) >= query_len` and
    contains characters from `[A-Za-z-]`. Empty string when no match
    columns exist (caller filters those rows out upstream).
    """
    slots = ["-"] * query_len
    # `inserts_before[i]` is the run of lowercase residues that would
    # land just before slot `i` (equivalently, just after slot `i-1`).
    # `inserts_before[0]` holds leading inserts, `inserts_before[query_len]`
    # trailing.
    inserts_before: list[list[str]] = [[] for _ in range(query_len + 1)]
    last_qi = -1  # most recent query-side cursor we've seen
    q_first: int | None = None
    q_last: int | None = None

    for qi, ti in columns:
        if qi >= 0 and ti >= 0:
            if 0 <= qi < query_len and 0 <= ti < len(target_seq):
                slots[qi] = target_seq[ti].upper()
                if q_first is None:
                    q_first = qi
                q_last = qi
            last_qi = qi
        elif qi >= 0 and ti < 0:
            # Target-side gap on a query column — advance the cursor
            # so subsequent inserts are positioned correctly.
            last_qi = qi
        elif qi < 0 and ti >= 0:
            if 0 <= ti < len(target_seq):
                pos = max(0, min(last_qi + 1, query_len))
                inserts_before[pos].append(target_seq[ti].lower())

    if q_first is None or q_last is None:
        # No match columns — empty row. Caller drops these targets.
        return ""

    pieces: list[str] = []
    for qi in range(query_len):
        pieces.append(slots[qi])
        # `inserts_before[qi + 1]` is the run that lands after slot qi.
        # Keep only when both bounds are inside the matched span:
        # `q_first <= qi < q_last`. Equivalent: after a slot at-or-after
        # the first match, and before the last-match slot itself.
        if q_first <= qi < q_last:
            pieces.extend(inserts_before[qi + 1])
    return "".join(pieces)


def _normalize(s: str) -> str:
    """Strip whitespace, uppercase, drop gap characters."""
    return "".join(s.split()).upper().replace("-", "")


def _is_uppercase_alpha(s: str) -> bool:
    """All characters are A..Z. Empty string is False — that case is
    handled separately upstream so it gets its own error message."""
    if not s:
        return False
    return all("A" <= c <= "Z" for c in s)


__all__ = [
    "PairwiseAlignConfig",
    "PairwiseAlignOrchestrator",
    "PairwiseAlignRequest",
    "PairwiseAlignResult",
    "PairwiseAlignTarget",
    "PairwiseAlignment",
]
