"""Templates re-alignment orchestrator — unit tests with stubbed services.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.5. Pure-Python tests. The
embedding and align HTTP calls are replaced by an in-process stub
transport that returns canned shapes; no httpx, no model load, no
running services.

Covers the orchestrator-side behaviors the helper tests can't:

  - query normalization + length-cap enforcement
  - query/a3m length cross-check (caller paired the wrong query)
  - sanity-failed records dropped, job survives
  - dedup of identical template residues — embed call sees exactly
    one entry per unique sequence
  - records OTalign couldn't place (no match columns) are dropped
    from the output but stats reflect the drop
  - empty / single-record / one-residue inputs
  - max_records / max_query_length yield typed errors
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.templates import (
    TemplatesRealignConfig,
    TemplatesRealignOrchestrator,
    TemplatesRealignRequest,
)


# ---------------------------------------------------------------------------
# Stub transport — records calls, returns canned shapes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _EmbedCall:
    model: str
    sequences: list[str]


@dataclass(slots=True)
class _AlignCall:
    aligner: str
    mode: str
    n_targets: int
    options: dict[str, Any]


@dataclass(slots=True)
class StubTransport:
    """In-process stub for `TemplatesTransport`.

    By default produces a simple per-position embedding (`emb[i] = [i, len, hash]`)
    for each sequence and a diagonal alignment for each (query, target)
    pair. Tests override either behavior by setting a custom callable
    on `embed_fn` / `align_fn`.
    """

    embed_calls: list[_EmbedCall] = field(default_factory=list)
    align_calls: list[_AlignCall] = field(default_factory=list)
    embed_fn: Any = None
    align_fn: Any = None

    async def embed(
        self, *, model: str, sequences: Sequence[str]
    ) -> list[np.ndarray]:
        self.embed_calls.append(_EmbedCall(model=model, sequences=list(sequences)))
        if self.embed_fn is not None:
            return self.embed_fn(model, list(sequences))
        # Default: trivial embeddings — one (3,)-vector per residue, distinct
        # per (sequence, position) so length consistency tests pass.
        out: list[np.ndarray] = []
        for s_idx, seq in enumerate(sequences):
            arr = np.zeros((len(seq), 3), dtype=np.float32)
            for i, c in enumerate(seq):
                arr[i, 0] = float(ord(c))
                arr[i, 1] = float(i)
                arr[i, 2] = float(s_idx)
            out.append(arr)
        return out

    async def align(
        self,
        *,
        aligner: str,
        mode: str,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.align_calls.append(
            _AlignCall(
                aligner=aligner, mode=mode, n_targets=len(target_embeddings),
                options=dict(options),
            )
        )
        if self.align_fn is not None:
            return self.align_fn(query_embedding, list(target_embeddings))
        # Default: diagonal alignment from (0, 0) onwards, length =
        # min(Lq, Lt). Mirrors what a perfect glocal substring match
        # would produce.
        Lq = query_embedding.shape[0]
        out: list[dict[str, Any]] = []
        for t in target_embeddings:
            Lt = t.shape[0]
            n = min(Lq, Lt)
            cols = [[i, i] for i in range(n)]
            out.append({
                "score": 0.5,
                "mode": mode,
                "query_start": 0,
                "query_end": n,
                "target_start": 0,
                "target_end": n,
                "columns": cols,
            })
        return out


def _make(config: TemplatesRealignConfig | None = None) -> tuple[
    TemplatesRealignOrchestrator, StubTransport
]:
    transport = StubTransport()
    orch = TemplatesRealignOrchestrator(
        config=config or TemplatesRealignConfig(),
        transport=transport,
    )
    return orch, transport


def _a3m(*records: tuple[str, int, int, str]) -> str:
    """Build a small a3m text from `(target_id, start, end, row)` tuples."""
    lines: list[str] = []
    for target_id, start, end, row in records:
        lines.append(f">{target_id}/{start}-{end}")
        lines.append(row)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_two_records() -> None:
    """Diagonal stub alignment + identical-substring templates produce
    a valid output A3M with expected stats."""
    orch, transport = _make()
    query = "ABCDEFGHIJ"  # 10 residues, all uppercase A..Z
    text = _a3m(
        ("t1", 1, 5, "ABCDE-----"),  # upper=5 + gap=5 = 10 ✓; interval 5 ✓
        ("t2", 2, 6, "-FGHIJ----"),  # upper=5 + gap=5 = 10 ✓; interval 5 ✓
    )

    result = await orch.run(
        TemplatesRealignRequest(query_id="Q", query_sequence=query, a3m=text)
    )

    # One embed call covering query + 2 unique templates.
    assert len(transport.embed_calls) == 1
    call = transport.embed_calls[0]
    assert call.model == "ankh_large"
    assert call.sequences[0] == query
    assert len(call.sequences) == 3  # query + 2 unique templates

    # One align call with 2 targets.
    assert len(transport.align_calls) == 1
    assert transport.align_calls[0].mode == "glocal"
    assert transport.align_calls[0].aligner == "otalign"
    assert transport.align_calls[0].n_targets == 2

    # Output A3M is just the 2 hit rows — no query record (the input
    # hmmsearch a3m doesn't have one, and the output mirrors the
    # input shape).
    lines = result.payload.splitlines()
    assert lines[0].startswith(">t1/")
    assert lines[1].startswith("ABCDE")  # diagonal stub places residues at positions 0..4
    assert lines[2].startswith(">t2/")
    assert lines[3].startswith("FGHIJ")
    assert all(len(row) == len(query) for row in (lines[1], lines[3]))

    assert result.stats["records_kept"] == 2
    assert result.stats["records_dropped_sanity"] == 0
    assert result.stats["records_dropped_no_match"] == 0
    assert result.stats["unique_template_seqs"] == 2


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_normalized_upper_and_gaps_stripped() -> None:
    orch, transport = _make()
    text = _a3m(("t", 1, 5, "ABCDE-----"))
    # Caller passes lowercase, gaps, leading/trailing whitespace.
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="  abc-de--fghij  ",
            a3m=text,
        )
    )
    assert transport.embed_calls[0].sequences[0] == "ABCDEFGHIJ"
    assert result.stats["query_length"] == 10
    # Query is normalized for embedding but does NOT appear in the
    # output payload (output mirrors hmmsearch shape).
    assert "ABCDEFGHIJ" not in result.payload.splitlines()


@pytest.mark.asyncio
async def test_empty_query_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(query_id="Q", query_sequence="", a3m="")
        )
    assert excinfo.value.code == ErrorCode.INVALID_FASTA


@pytest.mark.asyncio
async def test_query_with_non_alphabetic_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(
                query_id="Q",
                query_sequence="ABC123",
                a3m=_a3m(("t", 1, 3, "ABC")),
            )
        )
    assert excinfo.value.code == ErrorCode.INVALID_FASTA


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_too_long_raises() -> None:
    config = TemplatesRealignConfig(max_query_length=5)
    orch, _ = _make(config)
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(
                query_id="Q",
                query_sequence="ABCDEFGH",
                a3m=_a3m(("t", 1, 8, "ABCDEFGH")),
            )
        )
    assert excinfo.value.code == ErrorCode.SEQ_TOO_LONG
    assert excinfo.value.http_status == 400


@pytest.mark.asyncio
async def test_template_too_long_raises() -> None:
    """Even when the query fits, an oversized template fails the
    length cap (PLAN §1.2 — every PLM enforces the same limit).

    Row design — query_len=5, template residues=8:
      "ABCDEfff" → upper=5, lower=3, gap=0
      match_cols (upper+gap) = 5 (matches query length)
      interval (upper+lower) = 8 (header /1-8)
    """
    config = TemplatesRealignConfig(max_query_length=5)
    orch, _ = _make(config)
    text = _a3m(("t", 1, 8, "ABCDEfff"))
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(query_id="Q", query_sequence="ABCDE", a3m=text)
        )
    assert excinfo.value.code == ErrorCode.SEQ_TOO_LONG
    assert excinfo.value.http_status == 400


@pytest.mark.asyncio
async def test_too_many_records_raises_queue_full() -> None:
    config = TemplatesRealignConfig(max_records=2)
    orch, _ = _make(config)
    text = _a3m(
        ("t1", 1, 3, "ABC"),
        ("t2", 1, 3, "DEF"),
        ("t3", 1, 3, "GHI"),
    )
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(query_id="Q", query_sequence="ABC", a3m=text)
        )
    assert excinfo.value.code == ErrorCode.QUEUE_FULL
    assert excinfo.value.http_status == 413


# ---------------------------------------------------------------------------
# Query / a3m mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_length_mismatch_with_a3m_raises() -> None:
    """Caller paired the wrong query with the a3m — the parser drops
    every record under `query_len_mismatch:`, and the orchestrator
    elevates that into a typed error."""
    orch, _ = _make()
    text = _a3m(("t", 1, 5, "ABCDE"))  # implies query_len=5
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(
                query_id="Q",
                query_sequence="ABCDEFGHIJ",  # 10
                a3m=text,
            )
        )
    assert excinfo.value.code == ErrorCode.INVALID_FASTA
    assert "match-state count" in excinfo.value.message


@pytest.mark.asyncio
async def test_a3m_with_no_records_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as excinfo:
        await orch.run(
            TemplatesRealignRequest(query_id="Q", query_sequence="ABC", a3m="")
        )
    assert excinfo.value.code == ErrorCode.INVALID_FASTA


# ---------------------------------------------------------------------------
# Drop accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanity_failed_records_drop_but_job_survives() -> None:
    """One bogus record out of three: stats reports the drop, output
    A3M contains 2 hit rows."""
    orch, _ = _make()
    text = _a3m(
        ("t1", 1, 5, "ABCDE"),
        # interval mismatch: row says 3 residues but header says 1-5
        ("bad", 1, 5, "AB-"),
        ("t2", 1, 5, "FGHIJ"),
    )
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="ABCDE",
            a3m=text,
        )
    )
    assert result.stats["records_in"] == 3
    assert result.stats["records_kept"] == 2
    assert result.stats["records_dropped_sanity"] == 1
    # Output: 2 hit rows = 4 lines (no query record at top).
    body = result.payload.splitlines()
    assert body[0].startswith(">t1/")
    assert body[2].startswith(">t2/")
    assert ">bad/" not in result.payload


@pytest.mark.asyncio
async def test_records_with_no_match_are_dropped_from_output() -> None:
    """Override the stub align to return zero-match alignments — the
    orchestrator drops those records from the output and counts them
    in `records_dropped_no_match`."""
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 0.0,
            "mode": "glocal",
            "query_start": 0,
            "query_end": 0,
            "target_start": 0,
            "target_end": 0,
            "columns": [],
        }
        for _ in ts
    ]
    text = _a3m(
        ("t1", 1, 3, "ABC"),
        ("t2", 1, 3, "DEF"),
    )
    result = await orch.run(
        TemplatesRealignRequest(query_id="Q", query_sequence="ABC", a3m=text)
    )
    assert result.stats["records_kept"] == 0
    assert result.stats["records_dropped_no_match"] == 2
    # No records survive → output is empty.
    assert result.payload == ""


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_of_identical_templates_embeds_once() -> None:
    """Two records with identical residues → embed call sees exactly
    one entry per unique sequence; both rows still get rendered."""
    orch, transport = _make()
    text = _a3m(
        ("t1", 10, 14, "ABCDE"),  # raw "ABCDE", interval 5
        ("t2", 20, 24, "ABCDE"),  # same raw seq
        ("t3", 30, 34, "FGHIJ"),  # different
    )
    result = await orch.run(
        TemplatesRealignRequest(query_id="Q", query_sequence="ABCDE", a3m=text)
    )
    # Embed call: query + 2 unique templates = 3 sequences.
    assert len(transport.embed_calls[0].sequences) == 3
    # Align call: still 3 records (full per-record fan-out).
    assert transport.align_calls[0].n_targets == 3
    # Output: 3 hit rows, headers re-intervalled to their respective starts.
    body = result.payload.splitlines()
    assert ">t1/10-" in body[0]
    assert ">t2/20-" in body[2]
    assert ">t3/30-" in body[4]
    assert result.stats["unique_template_seqs"] == 2


# ---------------------------------------------------------------------------
# Edge sizes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_record() -> None:
    orch, _ = _make()
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="ABC",
            a3m=_a3m(("t", 1, 3, "ABC")),
        )
    )
    body = result.payload.splitlines()
    assert body[0].startswith(">t/")
    assert body[1] == "ABC"


@pytest.mark.asyncio
async def test_one_residue_query_and_template() -> None:
    """Degenerate single-residue input — orchestrator still produces
    a well-formed A3M."""
    orch, _ = _make()
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="M",
            a3m=_a3m(("t", 1, 1, "M")),
        )
    )
    assert result.stats["records_kept"] == 1
    body = result.payload.splitlines()
    assert body == [">t/1-1 Score=0.500", "M"]


# ---------------------------------------------------------------------------
# Custom request fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_overrides_default_model_and_mode() -> None:
    """Per-request `model` / `mode` win over config defaults; both are
    passed straight through to the transport."""
    orch, transport = _make()
    await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="ABC",
            a3m=_a3m(("t", 1, 3, "ABC")),
            model="ankh_cl",
            mode="q2t",
            options={"eps": 0.05},
        )
    )
    assert transport.embed_calls[0].model == "ankh_cl"
    assert transport.align_calls[0].mode == "q2t"
    assert transport.align_calls[0].options == {"eps": 0.05}


# ---------------------------------------------------------------------------
# Output format invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_rows_have_no_lowercase() -> None:
    """The no-insertions rule (PLAN §2): every output row contains only
    uppercase residues and `-`.

    Row design — query_len=3, template residues=4:
      "ABcD" → upper=3, lower=1, gap=0
      match_cols = 3 (matches query length)
      interval = 4 (header /1-4)
      raw_seq = "ABCD"
    """
    orch, transport = _make()
    # Stub align: interleave a (-1, ti) insert column among matches.
    transport.align_fn = lambda q, ts: [
        {
            "score": 0.5,
            "mode": "glocal",
            "query_start": 0,
            "query_end": q.shape[0],
            "target_start": 0,
            "target_end": t.shape[0],
            "columns": [
                [0, 0],
                [-1, 1],
                [1, 2],
                [2, 3],
            ],
        }
        for t in ts
    ]
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="ABC",
            a3m=_a3m(("t", 1, 4, "ABcD")),
        )
    )
    body = result.payload.splitlines()
    # body[0] is the header, body[1] is the hit row.
    hit_row = body[1]
    assert len(hit_row) == 3
    # Only uppercase or `-` allowed.
    assert all(c == "-" or c.isupper() for c in hit_row)


@pytest.mark.asyncio
async def test_default_preserves_input_order() -> None:
    """Default `sort_by_score=False` preserves input record order
    regardless of per-record scores."""
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        # Score the records in *reverse* of their input order so
        # input-order != score-order. With the default we still emit
        # in input order.
        {
            "score": 0.1 * (len(ts) - i),
            "mode": "glocal",
            "query_start": 0, "query_end": 1,
            "target_start": 0, "target_end": 1,
            "columns": [[0, 0]],
        }
        for i, _ in enumerate(ts)
    ]
    text = _a3m(
        ("first", 1, 1, "A"),
        ("second", 1, 1, "B"),
        ("third", 1, 1, "C"),
    )
    result = await orch.run(
        TemplatesRealignRequest(query_id="Q", query_sequence="A", a3m=text)
    )
    headers = [ln for ln in result.payload.splitlines() if ln.startswith(">")]
    assert headers == [
        ">first/1-1 Score=0.300",
        ">second/1-1 Score=0.200",
        ">third/1-1 Score=0.100",
    ]
    assert result.stats["sort_by_score"] is False


@pytest.mark.asyncio
async def test_sort_by_score_emits_best_hit_first() -> None:
    """`sort_by_score=True` orders records descending by OTalign score."""
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 0.1 * (len(ts) - i),
            "mode": "glocal",
            "query_start": 0, "query_end": 1,
            "target_start": 0, "target_end": 1,
            "columns": [[0, 0]],
        }
        for i, _ in enumerate(ts)
    ]
    text = _a3m(
        ("first", 1, 1, "A"),
        ("second", 1, 1, "B"),
        ("third", 1, 1, "C"),
    )
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="A",
            a3m=text,
            sort_by_score=True,
        )
    )
    headers = [ln for ln in result.payload.splitlines() if ln.startswith(">")]
    # In this stub, input order is already score-descending
    # (first=0.3, second=0.2, third=0.1) so a sort is a no-op. The
    # dedicated reorder test below pins a non-trivial reorder.
    assert headers == [
        ">first/1-1 Score=0.300",
        ">second/1-1 Score=0.200",
        ">third/1-1 Score=0.100",
    ]
    assert result.stats["sort_by_score"] is True


@pytest.mark.asyncio
async def test_sort_by_score_actually_reorders() -> None:
    """Non-trivial reorder: highest-scoring record is *not* first in
    the input, so sort changes the row order."""
    orch, transport = _make()
    scores = {0: 0.1, 1: 0.9, 2: 0.5}  # second record wins
    transport.align_fn = lambda q, ts: [
        {
            "score": scores[i],
            "mode": "glocal",
            "query_start": 0, "query_end": 1,
            "target_start": 0, "target_end": 1,
            "columns": [[0, 0]],
        }
        for i in range(len(ts))
    ]
    text = _a3m(
        ("a", 1, 1, "A"),
        ("b", 1, 1, "B"),
        ("c", 1, 1, "C"),
    )
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="A",
            a3m=text,
            sort_by_score=True,
        )
    )
    headers = [ln for ln in result.payload.splitlines() if ln.startswith(">")]
    assert headers == [
        ">b/1-1 Score=0.900",
        ">c/1-1 Score=0.500",
        ">a/1-1 Score=0.100",
    ]


@pytest.mark.asyncio
async def test_header_reintervals_with_target_span() -> None:
    """Header `/start-end` follows OTalign's actually-placed template
    span (PLAN §2). Stub aligns target indices 1..3 of a 5-residue
    template → new interval = `orig_start+1 .. orig_start+3`.

    Row design — query_len=3, template residues=5:
      "ABCab" → upper=3, lower=2, gap=0
      match_cols = 3 (matches query length)
      interval = 5 (header /100-104)
      raw_seq = "ABCAB" (lowercase a/b uppercased)
    """
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 0.75,
            "mode": "glocal",
            "query_start": 0,
            "query_end": 3,
            "target_start": 1,
            "target_end": 4,
            "columns": [[0, 1], [1, 2], [2, 3]],
        }
        for _ in ts
    ]
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="Q",
            query_sequence="ABC",
            a3m=_a3m(("t", 100, 104, "ABCab")),
        )
    )
    header = result.payload.splitlines()[0]
    # span = (1, 3) → new_start = 100+1 = 101; new_end = 100+3 = 103.
    assert header.startswith(">t/101-103 ")
    assert "Score=0.750" in header
