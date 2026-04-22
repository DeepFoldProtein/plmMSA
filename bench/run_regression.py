#!/usr/bin/env python3
"""Live regression harness for the plmMSA stack.

Runs each CASP15 fixture through the real `api:/v2/msa` endpoint, polls the
job until it terminates, and compares the produced MSA to the baseline
numbers in `tests/fixtures/casp15/expected_stats.json`. Not run by CI — the
stack must be up, tokens set, and the sequence cache populated.

Usage::

    ADMIN_TOKEN=$(grep '^ADMIN_TOKEN=' .env | cut -d= -f2-) \
        uv run python bench/run_regression.py \
        --api-url http://localhost:8080 \
        [--targets T1104,T1120] \
        [--poll-interval 5] \
        [--poll-timeout 600]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "casp15"


@dataclass(slots=True)
class TargetSpec:
    name: str
    query_seq: str
    expected: dict[str, Any]


def load_targets(filter_names: set[str] | None = None) -> list[TargetSpec]:
    data = json.loads((_FIXTURE_DIR / "expected_stats.json").read_text())
    out: list[TargetSpec] = []
    for name, expected in data["targets"].items():
        if filter_names and name not in filter_names:
            continue
        fasta_text = (_FIXTURE_DIR / f"{name}.fasta").read_text()
        seq = "".join(ln.strip() for ln in fasta_text.splitlines() if ln and not ln.startswith(">"))
        out.append(TargetSpec(name=name, query_seq=seq, expected=expected))
    return sorted(out, key=lambda t: t.name)


def submit_and_wait(
    client: httpx.Client,
    api_url: str,
    token: str,
    target: TargetSpec,
    poll_interval: float,
    poll_timeout: float,
    collection: str,
    k: int,
    aligner: str,
    mode: str,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    submit = client.post(
        f"{api_url}/v2/msa",
        headers=headers,
        json={
            "sequences": [target.query_seq],
            "query_id": target.name,
            "model": "ankh_cl",
            "collection": collection,
            "k": k,
            "aligner": aligner,
            "mode": mode,
        },
    )
    submit.raise_for_status()
    job_id = submit.json()["job_id"]

    deadline = time.monotonic() + poll_timeout
    while True:
        resp = client.get(f"{api_url}/v2/msa/{job_id}", headers=headers)
        resp.raise_for_status()
        body = resp.json()
        status = body["status"]
        if status in {"succeeded", "failed", "cancelled"}:
            return body
        if time.monotonic() > deadline:
            raise TimeoutError(f"{target.name}: job {job_id} still {status} after {poll_timeout}s")
        time.sleep(poll_interval)


def evaluate(target: TargetSpec, job_result: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if job_result["status"] != "succeeded":
        issues.append(f"status={job_result['status']}; error={job_result.get('error')}")
        return False, issues

    result = job_result["result"]
    if result["format"] != "a3m":
        issues.append(f"unexpected format {result['format']!r}")

    a3m_lines = result["payload"].splitlines()
    depth = sum(1 for ln in a3m_lines if ln.startswith(">"))

    expected = target.expected
    tol = expected["tolerance"]
    target_depth = expected["msa_depth"]
    if abs(depth - target_depth) / max(target_depth, 1) > tol["depth_pct"] / 100:
        issues.append(f"depth drift: got {depth}, expected ~{target_depth} (±{tol['depth_pct']}%)")

    stats = result.get("stats") or {}
    if stats.get("pipeline") != "orchestrator":
        issues.append(f"pipeline={stats.get('pipeline')!r}; expected 'orchestrator'")

    return not issues, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live CASP15 regression harness.")
    parser.add_argument("--api-url", default="http://localhost:8080")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target subset (default: all in expected_stats.json).",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--poll-timeout", type=float, default=600.0)
    parser.add_argument("--token", default=os.environ.get("ADMIN_TOKEN", ""))
    parser.add_argument("--collection", default="ankh_uniref50")
    parser.add_argument("--k", type=int, default=50, help="FAISS neighbors per query.")
    parser.add_argument("--aligner", default="plmalign")
    parser.add_argument("--mode", default="local")
    args = parser.parse_args(argv)

    if not args.token:
        print("ERROR: ADMIN_TOKEN not set. Pass --token or export ADMIN_TOKEN.", file=sys.stderr)
        return 2

    filter_names = set(args.targets.split(",")) if args.targets else None
    targets = load_targets(filter_names)
    if not targets:
        print("ERROR: no targets selected.", file=sys.stderr)
        return 2

    print(f"Running regression on {len(targets)} targets against {args.api_url}")
    failures = 0
    with httpx.Client(timeout=args.poll_timeout + 30) as client:
        for target in targets:
            print(f"\n--- {target.name} ({len(target.query_seq)} residues) ---")
            try:
                job = submit_and_wait(
                    client,
                    args.api_url,
                    args.token,
                    target,
                    args.poll_interval,
                    args.poll_timeout,
                    args.collection,
                    args.k,
                    args.aligner,
                    args.mode,
                )
            except Exception as exc:
                print(f"  FAIL: {exc}")
                failures += 1
                continue
            ok, issues = evaluate(target, job)
            if ok:
                print("  ok")
            else:
                failures += 1
                for issue in issues:
                    print(f"  FAIL: {issue}")

    print(f"\n{len(targets) - failures}/{len(targets)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
