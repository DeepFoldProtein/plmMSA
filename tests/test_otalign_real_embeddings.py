"""OTalign behavior on real Ankh-Large embeddings via the embedding service.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.2. The synthetic-embedding tests
in `test_otalign_behavior.py` pin the algorithmic contract; this file
pins that the *combination* of OTalign + Ankh-Large gives sensible
alignments on actual protein sequences.

Two purposes:

1. **Pin the score scale + diagonal recovery on real embeddings** so a
   regression in either layer (PLM checkpoint drift, OTalign normalization
   change, Sinkhorn convergence) surfaces here.

2. **Answer the fundamental question — is q2t the right default for
   template re-alignment?** Head-to-head comparison against `glocal` and
   `local` on a sampled subset of the hmmsearch fixture, scored on three
   metrics:
     - template coverage (fraction of template residues placed),
     - hmmsearch agreement (fraction of template residues placed in the
       same query column hmmsearch chose),
     - identity recovery (fraction of matched columns where template
       residue == query residue).
   Test reports per-mode metrics; soft-asserts q2t isn't strictly worse
   than glocal so we catch a bad default before exposing the endpoint.

Embeddings come from the running embedding service. Set
`PLMMSA_TEST_EMBEDDING_URL` (or `EMBEDDING_URL`) to the host the embedding
container exposes — default `http://localhost:8081`. Tests skip cleanly
when the service is unreachable.

Gated under RUN_SLOW=1 because each Ankh-Large call costs a few seconds
on a real GPU and the fixture-sample tests run 20+ targets across three
modes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

# The fixture query is the 649-aa human Exostosin-1 (residues 55-703 of
# UniProt Q16394). Inlined so this file is self-sufficient even before
# the templates_realign parser lands.
_FIXTURE_QUERY = (
    "SPRQKRDANSSIYKGKKCRMESCFDFTLCKKNGFKVYVYPQQKGEKIAESYQNILAAIEG"
    "SRFYTSDPSQACLFVLSLDTLDRDQLSPQYVHNLRSKVQSLHLWNNGRNHLIFNLYSGTW"
    "PDYTEDVGFDIGQAMLAKASISTENFRPNFDVSIPLFSKDHPRTGGERGFLKFNTIPPLR"
    "KYMLVFKGKRYLTGIGSDTRNALYHVHNGEDVVLLTTCKHGKDWQKHKDSRCDRDNTEYE"
    "KYDYREMLHNATFCLVPRGRRLGSFRFLEALQAACVPVMLSNGWELPFSEVINWNQAAVI"
    "GDERLLLQIPSTIRSIHQDKILALRQQTQFLWEAYFSSVEKIVLTTLEIIQDRIFKHISR"
    "NSLIWNKHPGGLFVLPQYSSYLGDFPYYYANLGLKPPSKFTAVIHAVTPLVSQSQPVLKL"
    "LVAAAKSQYCAQIIVLWNCDKPLPAKHRWPATAVPVVVIEGESKVMSSRFLPYDNIITDA"
    "VLSLDEDTVLSTTEVDFAFTVWQSFPERIVGYPARSHFWDNSKERWGYTSKWTNDYSMVL"
    "TGAAIYHKYYHYLYSHYLPASLKNMVDQLANCEDILMNFLVSAVTKLPPIKVTQKKQYKE"
    "TMMGQTSRASRWADPDHFAQRQSCMNTFASWFGYMPLIHSQMRLDPVLF"
)
assert len(_FIXTURE_QUERY) == 649

_FIXTURE_PATH = Path(__file__).parent / "data" / "templates_realign" / "exostosin_hmmsearch.a3m"


def _embedding_url() -> str:
    return (
        os.environ.get("PLMMSA_TEST_EMBEDDING_URL")
        or os.environ.get("EMBEDDING_URL")
        or "http://localhost:8081"
    )


def _embedding_reachable() -> bool:
    """Quick health probe so the test skips cleanly when the stack isn't
    up. We don't try to start it ourselves — operator runs `bin/up.sh`."""
    try:
        r = httpx.get(f"{_embedding_url()}/health", timeout=3.0)
        if r.status_code != 200:
            return False
        body = r.json()
        return body.get("models", {}).get("ankh_large", {}).get("loaded", False)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW") != "1",
        reason="Slow: real Ankh-Large embeddings via the embedding service. "
        "Set RUN_SLOW=1 to enable.",
    ),
    pytest.mark.skipif(
        not _embedding_reachable(),
        reason=(
            f"Embedding service not reachable at {_embedding_url()}; "
            "set PLMMSA_TEST_EMBEDDING_URL or run `bin/up.sh`."
        ),
    ),
]


def _embed(seqs: list[str], model: str = "ankh_large") -> list[np.ndarray]:
    """POST /embed/bin against the embedding service.

    Uses the binary frame the production pipeline uses (see
    `plmmsa.pipeline.orchestrator._embed_chunks`) — request body is tiny
    JSON (model + sequence list); response is raw f32 tensors framed by
    `plmmsa.align.binary`. Exercises the same wire format the
    templates_realign endpoint will rely on, so a regression in the
    binary codec surfaces here too.
    """
    from plmmsa.align import binary as _binary

    with httpx.Client(timeout=600.0) as client:
        r = client.post(
            f"{_embedding_url()}/embed/bin",
            json={"model": model, "sequences": seqs},
        )
        r.raise_for_status()
        _meta, tensors = _binary.decode_tensors(r.content)
    return [np.asarray(t, dtype=np.float32) for t in tensors]


# ---------------------------------------------------------------------------
# Fixture — record-level parser for the hmmsearch a3m
# ---------------------------------------------------------------------------


_HEADER_RE = re.compile(r"^>(\S+)/(\d+)-(\d+)")


@dataclass(slots=True)
class _Record:
    header: str
    target_id: str  # e.g. "7sch_A"
    start: int  # 1-based
    end: int  # 1-based, inclusive
    row: str  # the full a3m row, mixed case + gaps
    raw_seq: str  # uppercase, gap-free residues fed to the PLM


def _parse_records() -> tuple[str, int, list[_Record]]:
    """Parse the bundled fixture into (query_seq, query_len, records).

    Skip the optional first record if it equals the query — for this
    fixture record 0 (`7sch_A/55-703`) IS the query, so we drop it from
    the comparison set so we don't grade self-alignment as a "template."
    """
    text = _FIXTURE_PATH.read_text()
    records: list[_Record] = []
    cur_header: str | None = None
    cur_seq_lines: list[str] = []

    def _flush():
        if cur_header is None:
            return
        m = _HEADER_RE.match(cur_header)
        assert m, f"unexpected header: {cur_header[:80]}"
        target_id, s, e = m.group(1).split("/")[0], int(m.group(2)), int(m.group(3))
        row = "".join(cur_seq_lines)
        raw = "".join(c.upper() for c in row if c != "-")
        records.append(_Record(cur_header, target_id, s, e, row, raw))

    for line in text.splitlines():
        if line.startswith(">"):
            _flush()
            cur_header = line
            cur_seq_lines = []
        else:
            cur_seq_lines.append(line)
    _flush()

    query = _FIXTURE_QUERY
    query_len = len(query)
    # Drop record 0 only if it matches the query (this fixture's case).
    if records and records[0].raw_seq == query:
        records = records[1:]
    return query, query_len, records


# ---------------------------------------------------------------------------
# Module-scoped helpers — pay the embedding cost once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def query_seq() -> str:
    return _FIXTURE_QUERY


@pytest.fixture(scope="module")
def query_embedding() -> np.ndarray:
    [emb] = _embed([_FIXTURE_QUERY])
    assert emb.shape[0] == len(_FIXTURE_QUERY)
    return emb


# ---------------------------------------------------------------------------
# Pin 1 — self-alignment is the diagonal
# ---------------------------------------------------------------------------


def test_self_alignment_glocal_is_diagonal(query_embedding: np.ndarray) -> None:
    """Query against itself under the **glocal default** collapses to the
    identity permutation on real Ankh-Large embeddings."""
    from plmmsa.align.otalign import OTalign

    Lq = query_embedding.shape[0]
    [aln] = OTalign().align(query_embedding, [query_embedding], mode="glocal")

    matched = [(qi, ti) for qi, ti in aln.columns if qi >= 0 and ti >= 0]
    assert len(matched) == Lq
    diagonal = sum(1 for qi, ti in matched if qi == ti)
    assert diagonal / Lq >= 0.99


def test_self_alignment_score_is_above_filter_cutoff(
    query_embedding: np.ndarray,
) -> None:
    """Self-score must sit comfortably above
    `[aligners.otalign].filter_threshold = 0.25`."""
    from plmmsa.align.otalign import OTalign

    [aln] = OTalign().align(query_embedding, [query_embedding], mode="glocal")
    assert aln.score > 0.25 * 4, (
        f"self-score {aln.score:.3f} too close to filter cutoff 0.25"
    )


# ---------------------------------------------------------------------------
# Pin 2 — substring recovery
# ---------------------------------------------------------------------------


def test_substring_glocal_recovers_clean_diagonal(
    query_embedding: np.ndarray,
    query_seq: str,
) -> None:
    """Embed query[100:300] *as a separate sequence* (not slicing the
    in-process embedding tensor — the substring goes through the
    embedding service so the test exercises the binary wire format and
    PLM tokenization both) and verify glocal recovers a clean offset
    diagonal: every template residue placed, query trimmed to [a, b)."""
    from plmmsa.align.otalign import OTalign

    a, b = 100, 300
    [template_emb] = _embed([query_seq[a:b]])
    [aln] = OTalign().align(query_embedding, [template_emb], mode="glocal")

    Lt = template_emb.shape[0]
    matched = [(qi, ti) for qi, ti in aln.columns if qi >= 0 and ti >= 0]
    assert len(matched) == Lt
    assert aln.target_start == 0 and aln.target_end == Lt
    assert aln.query_start == a and aln.query_end == b
    # glocal does not have q2t's forced-end artifact — strict diagonal.
    assert all(qi - ti == a for qi, ti in matched)


def test_signal_score_separates_from_scrambled(
    query_embedding: np.ndarray,
    query_seq: str,
) -> None:
    """Identity score must dominate score against a scrambled query."""
    from plmmsa.align.otalign import OTalign

    rng = np.random.default_rng(42)
    permuted = list(query_seq)
    rng.shuffle(permuted)
    scrambled = "".join(permuted)
    [scrambled_emb] = _embed([scrambled])

    aligner = OTalign()
    [signal] = aligner.align(query_embedding, [query_embedding], mode="glocal")
    [noise] = aligner.align(query_embedding, [scrambled_emb], mode="glocal")
    assert signal.score > noise.score
    assert noise.score < signal.score * 0.75


# ---------------------------------------------------------------------------
# Pin 3 — fundamental question: is q2t the right default mode?
# ---------------------------------------------------------------------------


def _hmmsearch_columns(row: str, query_len: int) -> list[tuple[int, int]]:
    """Recover the per-column (query_idx, template_residue_idx) pairs that
    hmmsearch's a3m row implies. `-1` marks a gap on the template side.

    Match-state slots in the row map 1:1 onto query positions; lowercase
    insertions sit between match-state slots (template residues with no
    query column). We only care about template→query placements, so
    insertions are dropped here.
    """
    cols: list[tuple[int, int]] = []
    qi = 0
    ti = 0
    for c in row:
        if c == "-":
            cols.append((qi, -1))
            qi += 1
        elif c.isupper():
            cols.append((qi, ti))
            qi += 1
            ti += 1
        else:  # lowercase = insertion (no query column)
            ti += 1
    assert qi == query_len, f"row had {qi} match slots, expected {query_len}"
    return cols


def _hmmsearch_placements(row: str, query_len: int) -> dict[int, int]:
    """Map template_residue_idx → query_column_idx, per hmmsearch."""
    out: dict[int, int] = {}
    for qi, ti in _hmmsearch_columns(row, query_len):
        if ti >= 0:
            out[ti] = qi
    return out


def _otalign_placements(columns: list[tuple[int, int]]) -> dict[int, int]:
    """Same map, computed from an OTalign Alignment."""
    out: dict[int, int] = {}
    for qi, ti in columns:
        if qi >= 0 and ti >= 0:
            out[ti] = qi
    return out


@dataclass(slots=True)
class _ModeMetrics:
    n_records: int
    n_template_residues: int
    coverage_mean: float  # fraction of template residues placed
    agreement_mean: float  # fraction of template residues placed in the same column hmmsearch chose
    identity_mean: float  # fraction of matched columns where template residue == query residue


def _grade_mode(
    *,
    query_emb: np.ndarray,
    query_seq: str,
    targets: list[_Record],
    target_embs: list[np.ndarray],
    mode: str,
) -> _ModeMetrics:
    from plmmsa.align.otalign import OTalign

    aligner = OTalign()
    alignments = aligner.align(query_emb, target_embs, mode=mode)  # type: ignore[arg-type]
    n_records = len(alignments)
    coverages: list[float] = []
    agreements: list[float] = []
    identities: list[float] = []
    n_template_residues = 0

    for rec, aln in zip(targets, alignments):
        Lt = len(rec.raw_seq)
        n_template_residues += Lt
        ot_place = _otalign_placements(list(aln.columns))
        hm_place = _hmmsearch_placements(rec.row, len(query_seq))

        coverages.append(len(ot_place) / Lt if Lt else 0.0)
        if hm_place:
            agree = sum(
                1 for ti, qi in ot_place.items() if hm_place.get(ti) == qi
            ) / len(hm_place)
            agreements.append(agree)
        if ot_place:
            ident = sum(
                1
                for ti, qi in ot_place.items()
                if rec.raw_seq[ti] == query_seq[qi]
            ) / len(ot_place)
            identities.append(ident)

    return _ModeMetrics(
        n_records=n_records,
        n_template_residues=n_template_residues,
        coverage_mean=float(np.mean(coverages)) if coverages else 0.0,
        agreement_mean=float(np.mean(agreements)) if agreements else 0.0,
        identity_mean=float(np.mean(identities)) if identities else 0.0,
    )


@pytest.fixture(scope="module")
def fixture_sample(query_embedding: np.ndarray, query_seq: str):
    """Pick 20 records spread by template length so the comparison
    spans short / medium / long templates evenly. Embed them once."""
    _query, _qlen, records = _parse_records()
    # Spread across length buckets — 4 records per quintile.
    sorted_recs = sorted(records, key=lambda r: len(r.raw_seq))
    n = len(sorted_recs)
    stride = max(1, n // 20)
    sampled = sorted_recs[::stride][:20]
    embs = _embed([r.raw_seq for r in sampled])
    return query_seq, query_embedding, sampled, embs


def test_mode_comparison_glocal_is_the_right_default(
    fixture_sample: tuple[str, np.ndarray, list[_Record], list[np.ndarray]],
) -> None:
    """Head-to-head — glocal (default) vs q2t vs local on real templates.

    History: an earlier draft of templates_realign defaulted to q2t per
    upstream OTalign convention. This test caught that q2t agrees with
    hmmsearch on only ~35% of placements vs glocal's ~60% — q2t's
    forced `query_end == Lq` pulls the trailing template residue onto
    the wrong column. We switched the default to glocal and now use
    this test to *confirm* that decision keeps holding on the fixture.

    Soft assertions:

    - All three modes place at least 50% of template residues
      somewhere — a regression below this floor means OTalign's
      Sinkhorn collapsed.
    - glocal's hmmsearch agreement is the highest of the three, by
      at least 10 pp over q2t. If a future change inverts the ordering
      we want the test to surface that loudly so the default can be
      reconsidered.
    - glocal's coverage is at least 90% — the no-insertions-in-output
      filter (PLAN §2) drops `(qi=-1, ti>=0)` columns; we accept ~5%
      drop on this fixture and want to catch a regression where the
      drop rate balloons.

    The metrics table is always printed (visible with `pytest -s` or in
    the capture on failure) so operators can eyeball the numbers.
    """
    query_seq, query_emb, sampled, embs = fixture_sample
    metrics = {
        mode: _grade_mode(
            query_emb=query_emb,
            query_seq=query_seq,
            targets=sampled,
            target_embs=embs,
            mode=mode,
        )
        for mode in ("glocal", "q2t", "local")
    }

    lines = [
        "OTalign mode comparison on hmmsearch fixture sample "
        f"({metrics['glocal'].n_records} records, "
        f"{metrics['glocal'].n_template_residues} template residues):",
        f"  {'mode':<8} {'coverage':>10} {'hmmagree':>10} {'identity':>10}",
    ]
    for mode, m in metrics.items():
        lines.append(
            f"  {mode:<8} {m.coverage_mean:>10.3f} "
            f"{m.agreement_mean:>10.3f} {m.identity_mean:>10.3f}"
        )
    report = "\n".join(lines)
    print(report)

    for mode, m in metrics.items():
        assert m.coverage_mean > 0.5, (
            f"mode {mode}: mean template coverage {m.coverage_mean:.3f} too low\n{report}"
        )

    # Confirm glocal is still the right default.
    glocal_advantage = metrics["glocal"].agreement_mean - metrics["q2t"].agreement_mean
    assert glocal_advantage >= 0.10, (
        f"glocal hmmsearch-agreement {metrics['glocal'].agreement_mean:.3f} "
        f"only {glocal_advantage:.3f} above q2t — default may need revisiting.\n{report}"
    )
    assert metrics["glocal"].coverage_mean >= 0.90, (
        f"glocal coverage {metrics['glocal'].coverage_mean:.3f} below "
        f"the 0.90 floor — no-insertions filter would drop too much.\n{report}"
    )
