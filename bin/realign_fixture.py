#!/usr/bin/env python3
"""Run the templates re-alignment orchestrator on the bundled fixture.

Writes the output A3M + a JSON stats blob to `./tmp/`. Useful as a
one-shot smoke test / sample output for the
[`docs/templates-realign.md`](../docs/templates-realign.md) walkthrough.

Usage:
    PLMMSA_TEST_EMBEDDING_URL=http://172.28.0.3:8081 \
    PLMMSA_TEST_ALIGN_URL=http://172.28.0.7:8083    \
        uv run python bin/realign_fixture.py

The two env vars default to `http://localhost:8081` / `:8083`. Override
when the embedding / align containers aren't published on the host.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


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
_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "data" / "templates_realign" / "exostosin_hmmsearch.a3m"
)
_OUT_DIR = REPO_ROOT / "tmp"
_OUT_A3M = _OUT_DIR / "exostosin_realigned.a3m"
_OUT_STATS = _OUT_DIR / "exostosin_realigned.stats.json"


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


async def _run() -> None:
    from plmmsa.templates import (
        HttpTransport,
        TemplatesRealignConfig,
        TemplatesRealignOrchestrator,
        TemplatesRealignRequest,
    )

    transport = HttpTransport(
        embedding_url=_embedding_url(),
        align_url=_align_url(),
        timeout_s=900.0,
    )
    orch = TemplatesRealignOrchestrator(
        config=TemplatesRealignConfig(),
        transport=transport,
    )

    text = _FIXTURE_PATH.read_text()
    print(f"Fixture: {_FIXTURE_PATH.relative_to(REPO_ROOT)} "
          f"({len(text)} bytes)")
    print(f"Embedding: {_embedding_url()}")
    print(f"Align:     {_align_url()}")

    t0 = time.perf_counter()
    result = await orch.run(
        TemplatesRealignRequest(
            query_id="exostosin_query",
            query_sequence=_FIXTURE_QUERY,
            a3m=text,
        )
    )
    elapsed = time.perf_counter() - t0
    print(f"Re-aligned in {elapsed:.1f} s")

    _OUT_DIR.mkdir(exist_ok=True)
    _OUT_A3M.write_text(result.payload)
    _OUT_STATS.write_text(json.dumps(result.stats, indent=2) + "\n")
    print(f"Wrote {_OUT_A3M.relative_to(REPO_ROOT)} "
          f"({len(result.payload)} bytes)")
    print(f"Wrote {_OUT_STATS.relative_to(REPO_ROOT)}")
    print()
    print("Stats:")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_run())
