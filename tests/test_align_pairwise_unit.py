"""Pairwise-align orchestrator — unit tests with stubbed services.

Mirrors `test_templates_pipeline_unit.py`: the embedding and align HTTP
calls are replaced by an in-process stub transport that returns canned
shapes; no httpx, no model load, no running services.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from plmmsa.align.pairwise import (
    PairwiseAlignConfig,
    PairwiseAlignOrchestrator,
    PairwiseAlignRequest,
    PairwiseAlignTarget,
)
from plmmsa.errors import ErrorCode, PlmMSAError

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
    embed_calls: list[_EmbedCall] = field(default_factory=list)
    align_calls: list[_AlignCall] = field(default_factory=list)
    embed_fn: Any = None
    align_fn: Any = None

    async def embed(self, *, model: str, sequences: Sequence[str]) -> list[np.ndarray]:
        self.embed_calls.append(_EmbedCall(model=model, sequences=list(sequences)))
        if self.embed_fn is not None:
            return self.embed_fn(model, list(sequences))
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
                aligner=aligner,
                mode=mode,
                n_targets=len(target_embeddings),
                options=dict(options),
            )
        )
        if self.align_fn is not None:
            return self.align_fn(query_embedding, list(target_embeddings))
        query_len = query_embedding.shape[0]
        out: list[dict[str, Any]] = []
        for t in target_embeddings:
            target_len = t.shape[0]
            n = min(query_len, target_len)
            cols = [[i, i] for i in range(n)]
            out.append(
                {
                    "score": 0.5,
                    "mode": mode,
                    "query_start": 0,
                    "query_end": n,
                    "target_start": 0,
                    "target_end": n,
                    "columns": cols,
                }
            )
        return out


def _make(
    config: PairwiseAlignConfig | None = None,
) -> tuple[PairwiseAlignOrchestrator, StubTransport]:
    transport = StubTransport()
    orch = PairwiseAlignOrchestrator(
        config=config or PairwiseAlignConfig(),
        transport=transport,
    )
    return orch, transport


def _t(target_id: str, sequence: str) -> PairwiseAlignTarget:
    return PairwiseAlignTarget(target_id=target_id, sequence=sequence)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_two_targets() -> None:
    orch, transport = _make()
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABCDEFGHIJ",
            targets=[_t("t1", "ABCDEFG"), _t("t2", "FGHIJ")],
        )
    )

    # One embed call covering query + 2 unique targets.
    assert len(transport.embed_calls) == 1
    call = transport.embed_calls[0]
    assert call.model == "ankh_large"  # config default
    assert call.sequences[0] == "ABCDEFGHIJ"
    assert call.sequences[1:] == ["ABCDEFG", "FGHIJ"]

    # One align call fanned over 2 targets.
    assert len(transport.align_calls) == 1
    assert transport.align_calls[0].aligner == "otalign"
    assert transport.align_calls[0].mode == "glocal"
    assert transport.align_calls[0].n_targets == 2

    # Two alignments returned in input order, with diagonal columns from
    # the stub. A3M payload is populated by default — query at index 0,
    # then one header + row per kept target.
    assert [a.target_id for a in result.alignments] == ["t1", "t2"]
    assert all(a.score == 0.5 for a in result.alignments)
    assert result.a3m is not None
    a3m_lines = result.a3m.splitlines()
    assert a3m_lines[0] == ">q"
    assert a3m_lines[1] == "ABCDEFGHIJ"
    assert a3m_lines[2].startswith(">t1 score:0.500")
    assert a3m_lines[4].startswith(">t2 score:0.500")
    # The stub produces only match columns (no insertions), so each
    # target row is exactly query_len chars from [A-Z-]. The insert-
    # preserving renderer is exercised by a dedicated test below.
    query_len = 10
    for ln in (a3m_lines[3], a3m_lines[5]):
        assert len(ln) == query_len
        assert all(c == "-" or c.isupper() for c in ln)

    s = result.stats
    assert s["pipeline"] == "align_pairwise"
    assert s["targets_in"] == 2
    assert s["targets_kept"] == 2
    assert s["targets_dropped_no_match"] == 0
    assert s["unique_target_seqs"] == 2


# ---------------------------------------------------------------------------
# Query / target normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inputs_are_normalized_upper_and_gap_stripped() -> None:
    orch, transport = _make()
    await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="  abc-de  ",
            targets=[_t("t", "a-bcde")],
        )
    )
    assert transport.embed_calls[0].sequences == ["ABCDE", "ABCDE"]


@pytest.mark.asyncio
async def test_empty_query_raises_invalid_fasta() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(query_id="q", query_sequence="", targets=[_t("t", "ABC")])
        )
    assert exc.value.code == ErrorCode.INVALID_FASTA


@pytest.mark.asyncio
async def test_query_non_amino_acid_raises_invalid_fasta() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABC123",
                targets=[_t("t", "ABC")],
            )
        )
    assert exc.value.code == ErrorCode.INVALID_FASTA


@pytest.mark.asyncio
async def test_empty_targets_raises_invalid_fasta() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(PairwiseAlignRequest(query_id="q", query_sequence="ABC", targets=[]))
    assert exc.value.code == ErrorCode.INVALID_FASTA


@pytest.mark.asyncio
async def test_target_empty_after_normalization_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABC",
                targets=[_t("bad", "----")],
            )
        )
    assert exc.value.code == ErrorCode.INVALID_FASTA
    assert (exc.value.detail or {}).get("target_id") == "bad"


@pytest.mark.asyncio
async def test_target_non_amino_acid_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABC",
                targets=[_t("t", "AB1C")],
            )
        )
    assert exc.value.code == ErrorCode.INVALID_FASTA


@pytest.mark.asyncio
async def test_duplicate_target_id_raises() -> None:
    orch, _ = _make()
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABC",
                targets=[_t("t", "ABC"), _t("t", "DEF")],
            )
        )
    assert exc.value.code == ErrorCode.INVALID_FASTA
    assert "duplicate" in exc.value.message


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_too_long_raises_seq_too_long() -> None:
    orch, _ = _make(PairwiseAlignConfig(max_query_length=5))
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABCDEFGH",
                targets=[_t("t", "ABC")],
            )
        )
    assert exc.value.code == ErrorCode.SEQ_TOO_LONG
    assert exc.value.http_status == 400


@pytest.mark.asyncio
async def test_target_too_long_raises_seq_too_long() -> None:
    orch, _ = _make(PairwiseAlignConfig(max_query_length=5))
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABCDE",
                targets=[_t("oversized", "ABCDEFGH")],
            )
        )
    assert exc.value.code == ErrorCode.SEQ_TOO_LONG
    assert (exc.value.detail or {}).get("target_id") == "oversized"


@pytest.mark.asyncio
async def test_too_many_targets_raises_queue_full() -> None:
    orch, _ = _make(PairwiseAlignConfig(max_records=2))
    with pytest.raises(PlmMSAError) as exc:
        await orch.run(
            PairwiseAlignRequest(
                query_id="q",
                query_sequence="ABC",
                targets=[_t("a", "A"), _t("b", "B"), _t("c", "C")],
            )
        )
    assert exc.value.code == ErrorCode.QUEUE_FULL
    assert exc.value.http_status == 413


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_identical_target_sequences_dedup_in_embed() -> None:
    orch, transport = _make()
    await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABC",
            targets=[
                _t("t1", "ABC"),
                _t("t2", "ABC"),  # same sequence, different id
                _t("t3", "DEF"),
            ],
        )
    )
    # Embed call: query + 2 unique target sequences.
    assert len(transport.embed_calls[0].sequences) == 3
    # Align still runs for all 3 targets (full fan-out).
    assert transport.align_calls[0].n_targets == 3


# ---------------------------------------------------------------------------
# No-match drop accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_targets_with_no_match_dropped_from_output() -> None:
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
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABC",
            targets=[_t("t1", "ABC"), _t("t2", "DEF")],
        )
    )
    assert result.alignments == []
    assert result.stats["targets_kept"] == 0
    assert result.stats["targets_dropped_no_match"] == 2
    # A3M still has the query record at index 0 even when every target
    # is dropped — the query line is unconditional.
    assert result.a3m == ">q\nABC\n"


# ---------------------------------------------------------------------------
# A3M layout: query at index 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a3m_starts_with_query_record() -> None:
    """The output A3M is ColabFold/AlphaFold-shape — query at line 0,
    then one `>id` + row pair per kept target."""
    orch, _ = _make()
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="MYQUERY",
            query_sequence="abcde",  # lowercase + gaps to confirm it's normalized
            targets=[_t("t", "ABCDE")],
        )
    )
    assert result.a3m is not None
    lines = result.a3m.splitlines()
    assert lines[0] == ">MYQUERY"
    assert lines[1] == "ABCDE"
    assert lines[2].startswith(">t score:")
    assert lines[3] == "ABCDE"


# ---------------------------------------------------------------------------
# Insertion handling: trim both ends, keep middle lowercase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middle_insertions_preserved_ends_trimmed() -> None:
    """OTalign places target residues at query columns (uppercase),
    inside-the-span insertions as lowercase, and trims leading/trailing
    insertions.

    Query length = 5 (ABCDE), template residues = 9 (xxABCyyDEzz =
    "XXABCYYDEZZ" after upper). Designed alignment path:

       column kind     | (qi, ti)
       --------------- | --------
       leading insert  | (-1, 0)   # 'x' before any match → DROP
       leading insert  | (-1, 1)   # 'x' before any match → DROP
       match           | ( 0, 2)   # 'A' at query slot 0
       match           | ( 1, 3)   # 'B' at query slot 1
       match           | ( 2, 4)   # 'C' at query slot 2
       middle insert   | (-1, 5)   # 'y' between matches → KEEP (lowercase)
       middle insert   | (-1, 6)   # 'y' between matches → KEEP (lowercase)
       match           | ( 3, 7)   # 'D' at query slot 3
       match           | ( 4, 8)   # 'E' at query slot 4
       trailing insert | (-1, 9)   # 'z' after last match → DROP
       trailing insert | (-1,10)   # 'z' after last match → DROP

    Expected row: 'ABC' + 'yy' (lowercase middle inserts) + 'DE' = 'ABCyyDE'.
    Length = 5 (query) + 2 (kept inserts) = 7.
    """
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 1.0,
            "mode": "glocal",
            "query_start": 0,
            "query_end": 5,
            "target_start": 2,
            "target_end": 9,
            "columns": [
                [-1, 0],
                [-1, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [-1, 5],
                [-1, 6],
                [3, 7],
                [4, 8],
                [-1, 9],
                [-1, 10],
            ],
        }
        for _ in ts
    ]
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABCDE",
            targets=[_t("t", "xxABCyyDEzz")],
        )
    )
    row = result.alignments[0].a3m_row
    assert row == "ABCyyDE"
    # Sanity: uppercase count equals query_len (one per query column),
    # lowercase count equals kept inserts (2), total = 7.
    assert sum(1 for c in row if c.isupper()) == 5
    assert sum(1 for c in row if c.islower()) == 2
    assert len(row) == 7


@pytest.mark.asyncio
async def test_only_leading_and_trailing_inserts_trim_to_match_only_row() -> None:
    """If every insertion is outside the matched span, the row is the
    plain match-only string (no lowercase residues survive)."""
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 1.0,
            "mode": "glocal",
            "query_start": 0,
            "query_end": 3,
            "target_start": 1,
            "target_end": 4,
            "columns": [
                [-1, 0],  # leading insert → drop
                [0, 1],
                [1, 2],
                [2, 3],  # matches → uppercase slots
                [-1, 4],  # trailing insert → drop
            ],
        }
        for _ in ts
    ]
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABC",
            targets=[_t("t", "xABCx")],
        )
    )
    row = result.alignments[0].a3m_row
    assert row == "ABC"
    assert row is not None and not any(c.islower() for c in row)


# ---------------------------------------------------------------------------
# Output toggles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_a3m_false_skips_rendering() -> None:
    orch, _ = _make()
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABC",
            targets=[_t("t", "ABC")],
            emit_a3m=False,
        )
    )
    assert result.a3m is None
    assert result.alignments[0].a3m_row is None
    assert result.stats["emit_a3m"] is False


@pytest.mark.asyncio
async def test_default_preserves_input_order() -> None:
    orch, transport = _make()
    # Score targets in reverse-input order so input-order != score-order.
    transport.align_fn = lambda q, ts: [
        {
            "score": 0.1 * (len(ts) - i),
            "mode": "glocal",
            "query_start": 0,
            "query_end": 1,
            "target_start": 0,
            "target_end": 1,
            "columns": [[0, 0]],
        }
        for i, _ in enumerate(ts)
    ]
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="A",
            targets=[_t("first", "A"), _t("second", "B"), _t("third", "C")],
        )
    )
    assert [a.target_id for a in result.alignments] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_sort_by_score_actually_reorders() -> None:
    orch, transport = _make()
    scores = {0: 0.1, 1: 0.9, 2: 0.5}
    transport.align_fn = lambda q, ts: [
        {
            "score": scores[i],
            "mode": "glocal",
            "query_start": 0,
            "query_end": 1,
            "target_start": 0,
            "target_end": 1,
            "columns": [[0, 0]],
        }
        for i in range(len(ts))
    ]
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="A",
            targets=[_t("a", "A"), _t("b", "B"), _t("c", "C")],
            sort_by_score=True,
        )
    )
    assert [a.target_id for a in result.alignments] == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# Pass-through of model / mode / aligner / options
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_overrides_model_mode_aligner_options() -> None:
    orch, transport = _make()
    await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABC",
            targets=[_t("t", "ABC")],
            model="ankh_cl",
            mode="q2t",
            aligner="otalign",
            options={"eps": 0.05},
        )
    )
    assert transport.embed_calls[0].model == "ankh_cl"
    assert transport.align_calls[0].mode == "q2t"
    assert transport.align_calls[0].aligner == "otalign"
    assert transport.align_calls[0].options == {"eps": 0.05}


# ---------------------------------------------------------------------------
# Span pass-through (the orchestrator does not recompute spans)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spans_passed_through_from_align_service() -> None:
    orch, transport = _make()
    transport.align_fn = lambda q, ts: [
        {
            "score": 1.0,
            "mode": "glocal",
            "query_start": 2,
            "query_end": 5,
            "target_start": 1,
            "target_end": 4,
            "columns": [[2, 1], [3, 2], [4, 3]],
        }
        for _ in ts
    ]
    result = await orch.run(
        PairwiseAlignRequest(
            query_id="q",
            query_sequence="ABCDEFGH",
            targets=[_t("t", "WXYZ")],
        )
    )
    a = result.alignments[0]
    assert (a.query_start, a.query_end) == (2, 5)
    assert (a.target_start, a.target_end) == (1, 4)
