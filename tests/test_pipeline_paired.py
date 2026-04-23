"""Paired-MSA orchestrator tests.

Covers the MMseqs-style taxonomy join + paired A3M rendering using a
synthetic 2-chain fixture. The orchestrator's cross-PLM pipeline
(retrieval + fetch + align) is mocked via `client_factory` so these
tests exercise only the paired-specific code paths:
  - per-chain retrieval at `paired_k = multiplier * k` (capped)
  - taxonomy lookup via `TargetFetcher.fetch_taxonomy`
  - highest-scoring-per-chain-per-taxonomy selection
  - joint-score ranking
  - paired A3M gap-separator = max(chain_lens) // 10
  - empty-intersection degrades to query-only A3M
"""

from __future__ import annotations

import json
import struct

import httpx
import numpy as np
import pytest

from plmmsa.pipeline.a3m import AlignmentHit, assemble_paired_a3m
from plmmsa.pipeline.fetcher import DictTargetFetcher
from plmmsa.pipeline.orchestrator import Orchestrator, OrchestratorConfig
from plmmsa.pipeline.paired import join_by_taxonomy

# --- Unit tests on the pure-function join module -----------------------------


def _hit(tid: str, score: float) -> AlignmentHit:
    return AlignmentHit(
        target_id=tid,
        score=score,
        target_seq="M" * 5,
        columns=[(i, i) for i in range(5)],
    )


def test_join_shared_taxonomy_pairs_once() -> None:
    """Taxonomy A appears in both chains — emit one paired row."""
    res = join_by_taxonomy(
        per_chain_hits=[[_hit("h1", 10.0)], [_hit("h2", 20.0)]],
        per_chain_tax=[{"h1": "9606"}, {"h2": "9606"}],
    )
    assert len(res.rows) == 1
    assert res.rows[0].taxonomy_id == "9606"
    assert res.rows[0].joint_score == pytest.approx(30.0)
    assert res.shared_taxonomies == 1


def test_join_drops_taxonomies_missing_from_any_chain() -> None:
    """B only in chain 0 + C only in chain 1 → both dropped."""
    res = join_by_taxonomy(
        per_chain_hits=[
            [_hit("A0", 5.0), _hit("B0", 7.0)],
            [_hit("A1", 8.0), _hit("C1", 9.0)],
        ],
        per_chain_tax=[
            {"A0": "TaxA", "B0": "TaxB"},
            {"A1": "TaxA", "C1": "TaxC"},
        ],
    )
    assert [r.taxonomy_id for r in res.rows] == ["TaxA"]
    assert res.shared_taxonomies == 1
    # Bucket counts reflect all tax-annotated hits, not just the shared ones.
    assert res.taxonomies_per_chain == (2, 2)


def test_join_picks_best_hit_per_taxonomy_per_chain() -> None:
    """When a chain has multiple hits under the same taxonomy, the
    join keeps only the highest-scoring one in the paired row."""
    res = join_by_taxonomy(
        per_chain_hits=[
            [_hit("low", 3.0), _hit("high", 9.0)],
            [_hit("only", 5.0)],
        ],
        per_chain_tax=[
            {"low": "T", "high": "T"},
            {"only": "T"},
        ],
    )
    assert len(res.rows) == 1
    assert res.rows[0].hits[0].target_id == "high"
    assert res.rows[0].joint_score == pytest.approx(14.0)


def test_join_ranks_by_joint_score_descending() -> None:
    res = join_by_taxonomy(
        per_chain_hits=[
            [_hit("a0", 1.0), _hit("b0", 10.0)],
            [_hit("a1", 2.0), _hit("b1", 20.0)],
        ],
        per_chain_tax=[
            {"a0": "A", "b0": "B"},
            {"a1": "A", "b1": "B"},
        ],
    )
    assert [r.taxonomy_id for r in res.rows] == ["B", "A"]


def test_join_no_shared_taxonomy_returns_empty() -> None:
    res = join_by_taxonomy(
        per_chain_hits=[[_hit("x", 1.0)], [_hit("y", 2.0)]],
        per_chain_tax=[{"x": "A"}, {"y": "B"}],
    )
    assert res.rows == []
    assert res.shared_taxonomies == 0


def test_join_drops_hits_with_no_tax_record() -> None:
    """A hit whose id isn't in the tax map is excluded — treated as
    'taxonomy unknown'."""
    res = join_by_taxonomy(
        per_chain_hits=[[_hit("known", 5.0), _hit("unknown", 9.0)]],
        per_chain_tax=[{"known": "T"}],
    )
    # With N=1 chain + one taxonomy known, it pairs (trivial intersection).
    assert res.per_chain_in == (2,)
    assert res.per_chain_with_tax == (1,)


# --- Paired A3M renderer -----------------------------------------------------


def test_paired_a3m_gap_separator_is_max_len_div_10() -> None:
    """ColabFold convention — the separator between chains is a gap run
    of length `max(chain_lens) // 10`."""
    a3m = assemble_paired_a3m(
        query_ids=["A", "B"],
        query_seqs=["M" * 120, "K" * 40],  # max=120 → sep len=12
        paired_rows=[],
    )
    first_seq_line = a3m.splitlines()[1]
    assert first_seq_line == "M" * 120 + "-" * 12 + "K" * 40


def test_paired_a3m_empty_rows_still_emits_query() -> None:
    a3m = assemble_paired_a3m(
        query_ids=["A", "B"],
        query_seqs=["MK", "LI"],
        paired_rows=[],
    )
    lines = a3m.splitlines()
    assert lines[0].startswith(">A|B")
    assert "MK" in lines[1] and "LI" in lines[1]


# --- Orchestrator paired path ------------------------------------------------


def _encode_embed_bin(embeddings: list[np.ndarray]) -> bytes:
    """Mirror of `plmmsa.align.binary.encode_tensors` for the response
    body of /embed/bin used in this test's mocked embedding service."""
    meta = json.dumps({"model": "mock", "dim": int(embeddings[0].shape[-1])})
    meta_b = meta.encode("utf-8")
    parts: list[bytes] = [
        b"PLMA",
        struct.pack("<III", 1, len(meta_b), len(embeddings)),
        meta_b,
    ]
    for arr in embeddings:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        parts.append(struct.pack("<I", arr.ndim))
        parts.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def _encode_align_bin(alignments: list[dict]) -> bytes:
    body = {"alignments": alignments}
    raw = json.dumps(body).encode("utf-8")
    return raw


class _MockTransport(httpx.AsyncBaseTransport):
    """Serves just enough of /embed{,_bin}, /search, /align{,_bin} for
    the paired pipeline. Routes on URL path.
    """

    def __init__(
        self,
        *,
        chain_hits: dict[int, list[tuple[str, float]]],
        # Map from (sequence_prefix → chain_index) so we route per-chain
        # search results correctly. Chain 0 sequence starts with "AAA",
        # chain 1 with "BBB" — the test builds queries that way.
        chain_by_prefix: dict[str, int],
    ) -> None:
        self._chain_hits = chain_hits
        self._chain_by_prefix = chain_by_prefix
        # Track which query we're serving by matching /embed/bin input.
        self._last_chain_idx: int | None = None

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        url = request.url
        path = url.path
        if path == "/embed/bin":
            body = json.loads(request.content)
            seqs: list[str] = body["sequences"]
            # Record the chain index for the next /search call.
            if len(seqs) == 1 and seqs[0][:3] in self._chain_by_prefix:
                self._last_chain_idx = self._chain_by_prefix[seqs[0][:3]]
            # Emit dummy (Lq, dim=16) embeddings for each seq.
            embs = [np.ones((len(s), 16), dtype=np.float32) for s in seqs]
            return httpx.Response(
                200,
                content=_encode_embed_bin(embs),
                headers={"content-type": "application/x-plmmsa-embed"},
            )
        if path == "/search":
            body = json.loads(request.content)
            idx = self._last_chain_idx if self._last_chain_idx is not None else 0
            hits = self._chain_hits.get(idx, [])
            results = [[{"id": tid, "distance": 1.0 / (score + 1.0)} for tid, score in hits]]
            return httpx.Response(200, json={"results": results})
        if path == "/align/bin":
            # Parse the binary frame to count target tensors.
            content = request.content
            # Frame: magic(4) + version(4) + meta_len(4) + n_tensors(4) + ...
            (_magic, _ver, meta_len, n_tensors) = struct.unpack("<4sIII", content[:16])
            meta = json.loads(content[16 : 16 + meta_len].decode())
            idx = self._last_chain_idx if self._last_chain_idx is not None else 0
            hits = self._chain_hits.get(idx, [])
            # One alignment per target (n_tensors - 1, minus the query).
            alignments = [
                {
                    "score": hits[i][1] if i < len(hits) else 0.0,
                    "columns": [[0, 0], [1, 1]],
                }
                for i in range(n_tensors - 1)
            ]
            return httpx.Response(
                200,
                json={"alignments": alignments, "aligner": meta.get("aligner")},
            )
        return httpx.Response(404, text=f"unknown path: {path}")


def _make_orchestrator(
    *,
    chain_hits: dict[int, list[tuple[str, float]]],
    id_to_seq: dict[str, str],
    id_to_tax: dict[str, str],
    paired_k_multiplier: int = 3,
) -> Orchestrator:
    transport = _MockTransport(
        chain_hits=chain_hits,
        chain_by_prefix={"AAA": 0, "BBB": 1},
    )
    fetcher = DictTargetFetcher(id_to_seq, id_to_taxonomy=id_to_tax)

    def _client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=transport,
            base_url="http://mock",
        )

    cfg = OrchestratorConfig(
        embedding_url="http://mock",
        vdb_url="http://mock",
        align_url="http://mock",
        default_model="mock",
        default_aligner="plmalign",
        default_k=10,
        score_model="mock",
        paired_k_multiplier=paired_k_multiplier,
    )
    return Orchestrator(cfg, fetcher, client_factory=_client_factory)


async def test_run_paired_shared_taxonomy_produces_paired_row() -> None:
    orch = _make_orchestrator(
        chain_hits={
            0: [("hit0a", 50.0), ("hit0b", 20.0)],
            1: [("hit1a", 30.0), ("hit1b", 10.0)],
        },
        id_to_seq={
            "hit0a": "MKTIIAL",
            "hit0b": "MKTIIAL",
            "hit1a": "LIMKT",
            "hit1b": "LIMKT",
        },
        id_to_tax={
            "hit0a": "9606",
            "hit0b": "10090",
            "hit1a": "9606",
            "hit1b": "10090",
        },
    )
    result = await orch.run(
        {
            "sequences": ["AAAKTI", "BBBKTI"],
            "query_ids": ["chainA", "chainB"],
            "paired": True,
            "aligner": "plmalign",
            "score_model": "mock",
        }
    )
    assert result.format == "a3m"
    # Query + 2 paired rows (9606 and 10090 both appear in both chains).
    lines = [ln for ln in result.payload.splitlines() if ln.startswith(">")]
    assert lines[0].startswith(">chainA|chainB")
    tax_headers = [ln for ln in lines[1:] if ln.startswith(">tax:")]
    assert len(tax_headers) == 2
    # Joint scores: 9606 → 50+30=80 ranked first; 10090 → 20+10=30 second.
    assert ">tax:9606|hit0a|hit1a" in tax_headers[0]
    assert ">tax:10090|hit0b|hit1b" in tax_headers[1]
    # Stats block.
    assert result.stats["topology"] == "paired"
    assert result.stats["paired_rows"] == 2
    assert result.stats["paired_taxonomies"] == 2
    assert result.stats["paired_k_multiplier"] == 3
    assert result.stats["paired_k_effective"] == 30  # 3 * 10


async def test_run_paired_no_shared_taxonomy_returns_query_only() -> None:
    orch = _make_orchestrator(
        chain_hits={
            0: [("a0", 50.0)],
            1: [("b0", 30.0)],
        },
        id_to_seq={"a0": "MKT", "b0": "LIM"},
        id_to_tax={"a0": "TaxA", "b0": "TaxB"},
    )
    result = await orch.run(
        {
            "sequences": ["AAAKTI", "BBBKTI"],
            "paired": True,
            "score_model": "mock",
        }
    )
    # No shared tax → paired_rows=0, but query row is still emitted.
    lines = [ln for ln in result.payload.splitlines() if ln.startswith(">")]
    assert len(lines) == 1
    assert result.stats["paired_rows"] == 0
    assert result.stats["paired_taxonomies"] == 0


async def test_run_paired_respects_paired_k_multiplier() -> None:
    """With multiplier=5 and k=10, each chain should retrieve up to 50."""
    orch = _make_orchestrator(
        chain_hits={
            0: [(f"t{i}", 10.0) for i in range(50)],
            1: [(f"t{i}", 10.0) for i in range(50)],
        },
        id_to_seq={f"t{i}": "MK" for i in range(50)},
        id_to_tax={f"t{i}": "TaxA" for i in range(50)},
        paired_k_multiplier=5,
    )
    result = await orch.run(
        {
            "sequences": ["AAAKTI", "BBBKTI"],
            "paired": True,
            "k": 10,
            "score_model": "mock",
        }
    )
    assert result.stats["paired_k_effective"] == 50
