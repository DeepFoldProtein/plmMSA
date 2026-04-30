"""Templates re-alignment orchestrator.

End-to-end glue per `PLAN_TEMPLATES_REALIGN.md` §2. Takes a query
sequence + hmmsearch-style A3M, runs OTalign in glocal mode against
each template via Ankh-Large embeddings, and returns a re-rendered
A3M with re-intervalled headers and `Score=` stamps.

The orchestrator never touches HTTP directly — all I/O flows through
a `TemplatesTransport` (see `transport.py`). Tests inject a stub
transport; production wires `HttpTransport`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.templates.a3m_parser import (
    DroppedRecord,
    Record,
    parse_hmmsearch_a3m,
)
from plmmsa.templates.header import reinterval_header, stamp_score
from plmmsa.templates.render import kept_template_span, render_hit_match_only
from plmmsa.templates.transport import TemplatesTransport

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TemplatesRealignConfig:
    """Operator-tunable knobs for the orchestrator.

    `default_*` mirror the API edge defaults — callers can override per
    request via `TemplatesRealignRequest`. `max_query_length` and
    `max_records` are hard ceilings that produce typed errors when
    exceeded.
    """

    default_aligner: str = "otalign"
    default_model: str = "ankh_large"
    default_mode: str = "glocal"
    max_query_length: int = 1022  # min across PLMs (ESM1b's positional cap)
    max_records: int = 5000


@dataclass(slots=True)
class TemplatesRealignRequest:
    query_id: str
    query_sequence: str
    a3m: str
    model: str | None = None
    mode: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemplatesRealignResult:
    payload: str
    stats: dict[str, Any]
    format: str = "a3m"


class TemplatesRealignOrchestrator:
    """Single-shot orchestrator. Construct once per process; one
    instance handles many requests sequentially or concurrently
    (it carries no per-request state)."""

    def __init__(
        self,
        *,
        config: TemplatesRealignConfig,
        transport: TemplatesTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    async def run(self, request: TemplatesRealignRequest) -> TemplatesRealignResult:
        cfg = self._config

        # Step 1 — normalize query (PLAN §1.1).
        query_seq = "".join(request.query_sequence.split()).upper().replace("-", "")
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

        model = request.model or cfg.default_model
        mode = request.mode or cfg.default_mode
        aligner = cfg.default_aligner

        # Step 2 — parse a3m, cross-checked against the query length
        # (PLAN §1.3 — len(query_seq) must equal upper+gap of every record).
        parsed = parse_hmmsearch_a3m(request.a3m, expected_query_len=len(query_seq))
        records = parsed.records
        dropped = parsed.dropped

        if not records:
            self._raise_no_records(query_seq, dropped)

        # Step 3 — limits.
        for r in records:
            if len(r.raw_seq) > cfg.max_query_length:
                raise PlmMSAError(
                    f"template '{r.target_id}' length {len(r.raw_seq)} "
                    f"exceeds max {cfg.max_query_length}.",
                    code=ErrorCode.SEQ_TOO_LONG,
                    http_status=400,
                    detail={
                        "target_id": r.target_id,
                        "length": len(r.raw_seq),
                        "max": cfg.max_query_length,
                    },
                )
        if len(records) > cfg.max_records:
            raise PlmMSAError(
                f"a3m has {len(records)} records; max is {cfg.max_records}.",
                code=ErrorCode.QUEUE_FULL,
                http_status=413,
                detail={"records": len(records), "max": cfg.max_records},
            )

        # Step 4 — dedup unique template residue strings. Many hmmsearch
        # dumps repeat the same residue sequence under multiple PDB ids
        # (e.g. crystal-form variants of the same chain); embedding the
        # unique set saves real compute.
        seq_to_idx: dict[str, int] = {}
        unique_seqs: list[str] = []
        for r in records:
            if r.raw_seq not in seq_to_idx:
                seq_to_idx[r.raw_seq] = len(unique_seqs)
                unique_seqs.append(r.raw_seq)

        # Step 5 — embed query + unique templates in one call to the
        # transport (which chunks internally). Query lands at index 0.
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
                    "template embedding length does not match its sequence.",
                    code=ErrorCode.INTERNAL,
                    http_status=502,
                    detail={
                        "expected": len(seq),
                        "got": int(emb.shape[0]),
                    },
                )

        # Step 6 — fan templates back out from unique embeddings to the
        # per-record list (preserves input record order).
        target_embs = [unique_embs[seq_to_idx[r.raw_seq]] for r in records]

        # Step 7 — one batched align call.
        alignments = await self._transport.align(
            aligner=aligner,
            mode=mode,
            query_embedding=query_emb,
            target_embeddings=target_embs,
            options=dict(request.options),
        )
        if len(alignments) != len(records):
            raise PlmMSAError(
                "align service returned a different number of alignments than requested.",
                code=ErrorCode.INTERNAL,
                http_status=502,
                detail={"sent": len(records), "got": len(alignments)},
            )

        # Step 8 — for each record: drop (qi=-1) inserts, render row,
        # re-interval header, stamp score. Records where OTalign placed
        # zero template residues are dropped from the output (no row to
        # emit).
        out_records: list[tuple[Record, str, str]] = []
        records_dropped_no_match = 0
        for r, aln in zip(records, alignments, strict=True):
            cols = [(int(c[0]), int(c[1])) for c in aln.get("columns") or []]
            span = kept_template_span(cols)
            if span is None:
                records_dropped_no_match += 1
                continue
            new_start = r.start + span[0]
            new_end = r.start + span[1]
            score = float(aln.get("score", 0.0))
            new_row = render_hit_match_only(len(query_seq), r.raw_seq, cols)
            new_header = stamp_score(
                reinterval_header(r.header, new_start, new_end),
                score,
            )
            out_records.append((r, new_header, new_row))

        # Step 9 — assemble. Query record at top (no Score= — the query
        # has no alignment to itself in this pipeline; downstream tools
        # don't expect one).
        lines: list[str] = [f">{request.query_id}", query_seq]
        for _r, header, row in out_records:
            lines.append(header)
            lines.append(row)
        payload = "\n".join(lines) + "\n"

        stats = {
            "pipeline": "templates_realign",
            "query_length": len(query_seq),
            "records_in": len(records) + len(dropped),
            "records_kept": len(out_records),
            "records_dropped_sanity": len(dropped),
            "records_dropped_no_match": records_dropped_no_match,
            "unique_template_seqs": len(unique_seqs),
            "model": model,
            "mode": mode,
            "aligner": aligner,
        }
        return TemplatesRealignResult(payload=payload, stats=stats)

    def _raise_no_records(
        self,
        query_seq: str,
        dropped: list[DroppedRecord],
    ) -> None:
        """All input records were dropped — turn the most useful
        signal in the drop reasons into a typed error."""
        if not dropped:
            raise PlmMSAError(
                "a3m has no records.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
            )
        # If any record disagrees on query length, that is the headline
        # — the caller probably paired the wrong query with the a3m.
        for d in dropped:
            if d.reason.startswith("query_len_mismatch:"):
                raise PlmMSAError(
                    "query_sequence length does not match the a3m's "
                    "match-state count.",
                    code=ErrorCode.INVALID_FASTA,
                    http_status=400,
                    detail={
                        "query_length": len(query_seq),
                        "first_drop_reason": d.reason,
                    },
                )
        raise PlmMSAError(
            "a3m has no usable records (all dropped by sanity checks).",
            code=ErrorCode.INVALID_FASTA,
            http_status=400,
            detail={"drop_reasons": [d.reason for d in dropped[:5]]},
        )


def _is_uppercase_alpha(s: str) -> bool:
    """All characters are A..Z. Empty string is False — that case is
    handled separately upstream so it gets its own error message."""
    if not s:
        return False
    return all("A" <= c <= "Z" for c in s)


__all__ = [
    "TemplatesRealignConfig",
    "TemplatesRealignOrchestrator",
    "TemplatesRealignRequest",
    "TemplatesRealignResult",
]
