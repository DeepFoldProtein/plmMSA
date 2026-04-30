"""End-to-end templates re-alignment against the live stack.

Driven by `PLAN_TEMPLATES_REALIGN.md` §6.6. Runs the orchestrator
through `HttpTransport` against the running embedding + align services
on the fixture. Gated under `RUN_SLOW=1` because each run costs ~2 min
on real GPUs (~412 unique template embeds + one batched align over
593 targets).

Test asserts the output A3M's structural invariants — every row is
exactly `query_len` chars from `[A-Z-]`, no lowercase escapes the
no-insertions filter, kept-record coverage is ≥ 90% (matching the
§6.2 fixture-sample coverage). The numeric thresholds are pulled
from the same empirical run that motivated the glocal default.

Note on the parser invariant `upper+lower == end-start+1`: that rule
applies to *input* hmmsearch a3m. The orchestrator's output may have
`upper+lower < new_end-new_start+1` for records where OTalign dropped
interior template residues — see PLAN §2 trade-off. We do NOT
re-parse the output with the strict rule; we test the row-level
invariants directly.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import httpx
import pytest

from plmmsa.templates import (
    HttpTransport,
    TemplatesRealignConfig,
    TemplatesRealignOrchestrator,
    TemplatesRealignRequest,
    parse_hmmsearch_a3m,
)

# Same fixture query as test_otalign_real_embeddings.py.
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


def _align_url() -> str:
    return (
        os.environ.get("PLMMSA_TEST_ALIGN_URL")
        or os.environ.get("ALIGN_URL")
        or "http://localhost:8083"
    )


def _services_reachable() -> bool:
    """Probe both upstreams. Skip the test cleanly when either is down."""
    try:
        r = httpx.get(f"{_embedding_url()}/health", timeout=3.0)
        if r.status_code != 200:
            return False
        if not r.json().get("models", {}).get("ankh_large", {}).get("loaded"):
            return False
        r = httpx.get(f"{_align_url()}/health", timeout=3.0)
        if r.status_code != 200:
            return False
        if not r.json().get("aligners", {}).get("otalign", {}).get("loaded"):
            return False
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_SLOW") != "1",
        reason="Slow: full fixture run against live embedding + align "
        "services. Set RUN_SLOW=1 to enable.",
    ),
    pytest.mark.skipif(
        not _services_reachable(),
        reason=(
            f"embedding ({_embedding_url()}) or align ({_align_url()}) not "
            "reachable; set PLMMSA_TEST_EMBEDDING_URL / PLMMSA_TEST_ALIGN_URL "
            "or run `bin/up.sh`."
        ),
    ),
]


@pytest.mark.asyncio
async def test_full_fixture_runs_end_to_end() -> None:
    """Run the orchestrator on the 593-record Exostosin fixture and pin
    the structural invariants of the output A3M.
    """
    text = _FIXTURE_PATH.read_text()
    transport = HttpTransport(
        embedding_url=_embedding_url(),
        align_url=_align_url(),
        timeout_s=900.0,
    )
    config = TemplatesRealignConfig()
    orch = TemplatesRealignOrchestrator(config=config, transport=transport)

    result = await orch.run(
        TemplatesRealignRequest(
            query_id="exostosin_query",
            query_sequence=_FIXTURE_QUERY,
            a3m=text,
        )
    )

    stats = result.stats
    print(
        "templates_realign run:",
        {k: stats[k] for k in (
            "records_in",
            "records_kept",
            "records_dropped_sanity",
            "records_dropped_no_match",
            "unique_template_seqs",
        )},
    )

    # Structural sanity on the result object.
    assert result.format == "a3m"
    assert stats["query_length"] == len(_FIXTURE_QUERY)
    assert stats["records_in"] == 593
    assert stats["records_dropped_sanity"] == 0  # fixture is clean
    assert stats["records_kept"] >= int(0.90 * stats["records_in"]), (
        f"only {stats['records_kept']} / {stats['records_in']} records kept; "
        "OTalign placed nothing for too many records — the no-match drop is "
        "supposed to be < 10% on this fixture per §6.2"
    )

    # Output mirrors the hmmsearch a3m shape — no query record at top.
    lines = result.payload.splitlines()
    qlen = len(_FIXTURE_QUERY)
    headers: list[str] = []
    rows: list[str] = []
    for ln in lines:
        if ln.startswith(">"):
            headers.append(ln)
        else:
            rows.append(ln)
    assert len(headers) == len(rows) == stats["records_kept"]

    # Every row: exactly query_len chars, drawn from [A-Z-] only.
    bad: list[tuple[str, str]] = []
    for h, r in zip(headers, rows, strict=True):
        if len(r) != qlen or any(not (c == "-" or c.isupper()) for c in r):
            bad.append((h[:60], r[:30]))
    assert not bad, f"{len(bad)} row(s) violate the [A-Z-]^{qlen} invariant; first={bad[:3]}"

    # Every header: re-intervalled, score-stamped.
    for h in headers:
        assert "Score=" in h
        # First whitespace-separated token must match `>id/start-end`.
        first = h.split(maxsplit=1)[0]
        assert "/" in first
        prefix, span = first.rsplit("/", 1)
        s, e = span.split("-")
        assert int(s) <= int(e)


@pytest.mark.asyncio
async def test_identity_distribution_against_query() -> None:
    """Coarse identity-vs-query histogram over the kept rows.

    On real homologs the median identity at a matched query column
    sits in the 10-20% band (low absolute but well above random). This
    test pins the band so a regression in OTalign or Ankh-Large
    surfaces — not as a quality-of-MSA test (we don't grade MSAs here),
    just as a sanity floor.
    """
    text = _FIXTURE_PATH.read_text()
    transport = HttpTransport(
        embedding_url=_embedding_url(),
        align_url=_align_url(),
        timeout_s=900.0,
    )
    orch = TemplatesRealignOrchestrator(
        config=TemplatesRealignConfig(),
        transport=transport,
    )

    result = await orch.run(
        TemplatesRealignRequest(
            query_id="exostosin_query",
            query_sequence=_FIXTURE_QUERY,
            a3m=text,
        )
    )

    qlen = len(_FIXTURE_QUERY)
    rows = [
        ln
        for ln in result.payload.splitlines()
        if not ln.startswith(">")
    ]
    assert rows, "no rendered rows — earlier invariants should have caught this"

    # Per-row identity at matched query columns.
    identities: list[float] = []
    bucket = Counter[int]()
    for r in rows:
        matched = 0
        same = 0
        for qi in range(qlen):
            c = r[qi]
            if c == "-":
                continue
            matched += 1
            if c == _FIXTURE_QUERY[qi]:
                same += 1
        if matched == 0:
            continue
        ident = same / matched
        identities.append(ident)
        bucket[int(ident * 10)] += 1

    n = len(identities)
    median_ident = sorted(identities)[n // 2] if n else 0.0
    print(f"identity distribution over {n} rows; median = {median_ident:.3f}")
    print("  10%-bucket counts:", dict(sorted(bucket.items())))

    assert median_ident >= 0.10, (
        f"median identity {median_ident:.3f} below 0.10 — Ankh-Large "
        "embeddings or OTalign scoring may have drifted"
    )
