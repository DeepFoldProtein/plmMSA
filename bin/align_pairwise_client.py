#!/usr/bin/env python3
"""Example client for `POST /v2/align/pairwise`.

Reads a query FASTA + a targets FASTA from disk, posts them to the
plmMSA api service, and prints a per-target summary. Optionally writes
the rendered A3M and the raw JSON response to disk, and optionally
renders a score-histogram PNG.

The intent is a small, dependency-free reference --only the Python
stdlib is needed for the core flow (post + merge + write A3M). The
`--out-hist` flag lazy-imports `matplotlib`; install it only when you
want the histogram. The wire format is documented in
[`docs/align-pairwise-spec.md`].

Usage
-----
    export BASE_URL=https://plmmsa.deepfold.org   # default if unset

    python bin/align_pairwise_client.py \\
        --query     query.fasta          \\
        --targets   targets.fasta        \\
        --sort-by-score                  \\
        --out-a3m   aligned.a3m          \\
        --out-json  aligned.response.json

`/v2/align/pairwise` is the public surface -- no bearer token is
required against `plmmsa.deepfold.org`. Pass `--token` (or set
`$TOKEN` / `$ADMIN_TOKEN`) only when pointing at a gated deployment.

Large target lists --chunk + merge transparently. Each chunk is one
`POST /v2/align/pairwise`; the merged A3M carries one query header at
the top and all kept target rows in order (or score-desc when
`--sort-by-score` is set, sorted globally across chunks):

    python bin/align_pairwise_client.py \\
        --query        query.fasta      \\
        --targets      big_targets.fasta \\
        --chunk-size   2000             \\
        --sort-by-score                  \\
        --out-a3m      merged.a3m       \\
        --out-hist     score_hist.png

Or as a one-liner against the default public endpoint with no files
involved:

    python bin/align_pairwise_client.py \\
        --query-seq   MKTIIAL                 \\
        --target-seq  t1:MKTIIA               \\
        --target-seq  t2:AKTIIAL
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# FASTA reader --tiny enough to inline, no `biopython` required.
# ---------------------------------------------------------------------------


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Return `(id, sequence)` pairs from a FASTA file.

    `id` is the first whitespace-separated token after `>`. Sequence
    lines are concatenated verbatim; the server normalizes the residues
    (uppercase + drop gaps + drop whitespace) on its end, so we don't
    need to.
    """
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_seq: list[str] = []

    def flush() -> None:
        if current_id is not None:
            records.append((current_id, "".join(current_seq)))

    if str(path) == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(path).read_text().splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_id = line[1:].split()[0] if len(line) > 1 else ""
            current_seq = []
        else:
            current_seq.append(line)
    flush()
    return records


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Server returned a structured plmMSA error envelope."""

    def __init__(self, status: int, code: str, message: str, detail: object) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail


def post_align(
    *,
    base_url: str,
    token: str | None,
    body: dict,
    timeout_s: float,
) -> dict:
    url = base_url.rstrip("/") + "/v2/align/pairwise"
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # urllib's default UA ("Python-urllib/3.x") is on Cloudflare's
        # WAF banned-signature list -- deployments behind CF return
        # 403 1010 to it. Identify ourselves so the WAF lets us through.
        "User-Agent": "plmmsa-client/0.1 (+https://github.com/DeepFoldProtein/plmMSA)",
    }
    # /v2/align/pairwise is the public surface and does not require a
    # bearer token; only attach Authorization when the caller supplied
    # one (e.g. when pointing at an alternative gated deployment).
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Try to parse the plmMSA error envelope.
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise ApiError(e.code, "E_HTTP", raw, None) from None
        raise ApiError(
            e.code,
            payload.get("code", "E_UNKNOWN"),
            payload.get("message", raw),
            payload.get("detail"),
        ) from None


# ---------------------------------------------------------------------------
# Chunking + merge
# ---------------------------------------------------------------------------


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def run_chunked(
    *,
    base_url: str,
    token: str,
    query_id: str,
    query_sequence: str,
    targets: list[dict[str, str]],
    chunk_size: int,
    timeout_s: float,
    emit_a3m: bool,
    sort_by_score: bool,
    extra_body: dict,
    progress: bool = True,
) -> dict:
    """Post `targets` in chunks of `chunk_size` and merge the responses.

    Returns a single response-shape dict with `alignments`, `stats`, and
    (when `emit_a3m`) `a3m`. A `stats.chunks` key is added so callers
    can see the request was split.

    When `sort_by_score` is true the chunks are posted with
    `sort_by_score=false` and the merged alignments are sorted globally
    by score-desc afterwards; otherwise input order is preserved across
    chunks.
    """
    chunks = list(_chunked(targets, max(1, chunk_size)))
    n_chunks = len(chunks)

    merged_alignments: list[dict] = []
    base_stats: dict | None = None
    stats_sum = {
        "targets_in": 0,
        "targets_kept": 0,
        "targets_dropped_no_match": 0,
        "unique_target_seqs": 0,
    }
    query_block: str | None = None  # ">query_id\nresidues"

    t_total = time.time()
    for ci, (offset, chunk) in enumerate(chunks):
        body = {
            "query_id": query_id,
            "query_sequence": query_sequence,
            "targets": chunk,
            "emit_a3m": emit_a3m,
            # When chunked + sorting, sort globally after merge.
            "sort_by_score": sort_by_score and n_chunks == 1,
            **extra_body,
        }
        if progress:
            print(
                f"[chunk {ci + 1}/{n_chunks}] posting {len(chunk)} targets "
                f"({offset}-{offset + len(chunk) - 1})...",
                flush=True,
            )
        t0 = time.time()
        resp = post_align(base_url=base_url, token=token, body=body, timeout_s=timeout_s)
        dt = time.time() - t0
        stats = resp.get("stats") or {}
        if base_stats is None:
            base_stats = stats
        stats_sum["targets_in"] += int(stats.get("targets_in", 0))
        stats_sum["targets_kept"] += int(stats.get("targets_kept", 0))
        stats_sum["targets_dropped_no_match"] += int(stats.get("targets_dropped_no_match", 0))
        stats_sum["unique_target_seqs"] += int(stats.get("unique_target_seqs", 0))
        merged_alignments.extend(resp.get("alignments") or [])
        if emit_a3m and query_block is None:
            a3m_text = resp.get("a3m") or ""
            qb, _ = _split_a3m(a3m_text)
            if qb:
                query_block = qb
        if progress:
            print(
                f"  -> kept {stats.get('targets_kept')}/{stats.get('targets_in')}"
                f" (dropped {stats.get('targets_dropped_no_match')},"
                f" unique seqs {stats.get('unique_target_seqs')}) in {dt:.1f}s"
            )

    if sort_by_score and n_chunks > 1:
        merged_alignments.sort(key=lambda a: float(a.get("score", 0.0)), reverse=True)

    # Synthesize a merged response shape.
    merged_stats = dict(base_stats or {})
    merged_stats.update(stats_sum)
    merged_stats["sort_by_score"] = sort_by_score
    merged_stats["emit_a3m"] = emit_a3m
    merged_stats["chunks"] = n_chunks
    merged_stats["elapsed_s"] = round(time.time() - t_total, 3)

    merged: dict = {"alignments": merged_alignments, "stats": merged_stats}
    if emit_a3m:
        if query_block is None:
            # No chunk produced a query block (e.g. zero kept everywhere).
            query_block = f">{query_id}\n{query_sequence}"
        rows = []
        for a in merged_alignments:
            row = a.get("a3m_row")
            if row is None:
                continue
            rows.append(f">{a['target_id']} score:{float(a['score']):.3f}")
            rows.append(row)
        merged["a3m"] = "\n".join([query_block, *rows]) + ("\n" if rows else "\n")
    return merged


def _split_a3m(a3m_text: str) -> tuple[str, list[str]]:
    """Return (query_block, [target_blocks]) --each "block" is its
    header line + residue row joined by '\n'.
    """
    lines = [ln for ln in a3m_text.splitlines() if ln]
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.startswith(">"):
            if cur:
                blocks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    if not blocks:
        return "", []
    return "\n".join(blocks[0]), ["\n".join(b) for b in blocks[1:]]


# ---------------------------------------------------------------------------
# Histogram --optional, lazy matplotlib import
# ---------------------------------------------------------------------------


def write_histogram(
    *,
    scores: list[float],
    out_path: Path,
    bins: int,
    title: str | None,
) -> None:
    if not scores:
        raise SystemExit("--out-hist: no scores in response (nothing to plot).")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        missing = getattr(e, "name", "matplotlib") or "matplotlib"
        raise SystemExit(
            f"--out-hist needs matplotlib in this env (missing: {missing}). "
            f"Install with: pip install matplotlib"
        ) from None

    n = len(scores)
    s_sorted = sorted(scores)
    median = statistics.median(scores)
    mean = statistics.fmean(scores)
    q1, _, q3 = statistics.quantiles(scores, n=4) if n >= 4 else (median, median, median)
    p1 = s_sorted[max(0, int(0.01 * n))]
    p99 = s_sorted[min(n - 1, int(0.99 * n))]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(scores, bins=bins, color="#4C72B0", edgecolor="white")

    ax.axvline(median, color="#C44E52", linestyle="--", linewidth=1, label=f"median {median:.3f}")
    ax.axvline(q1, color="#55A868", linestyle=":", linewidth=1, label=f"Q1 {q1:.3f}")
    ax.axvline(q3, color="#55A868", linestyle=":", linewidth=1, label=f"Q3 {q3:.3f}")

    ax.set_title(title or "OTalign score histogram")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    stats_lines = [
        f"n = {n}",
        f"min = {s_sorted[0]:.3f}",
        f"max = {s_sorted[-1]:.3f}",
        f"mean = {mean:.3f}",
        f"p1 = {p1:.3f}",
        f"p99 = {p99:.3f}",
    ]
    ax.text(
        0.02,
        0.97,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_inline_target(spec: str) -> tuple[str, str]:
    """Parse a `--target-seq id:SEQUENCE` value."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--target-seq expects 'id:SEQUENCE', got {spec!r}")
    tid, seq = spec.split(":", 1)
    if not tid:
        raise argparse.ArgumentTypeError("--target-seq id is empty")
    if not seq:
        raise argparse.ArgumentTypeError(f"--target-seq sequence for {tid!r} is empty")
    return tid, seq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post a query + targets to /v2/align/pairwise and print a summary. "
            "Supports chunked POST (for target lists larger than the server's "
            "per-request cap) and an optional score histogram. "
            "See docs/align-pairwise-spec.md for the API contract."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BASE_URL", "https://plmmsa.deepfold.org"),
        help="api service base URL (default $BASE_URL or https://plmmsa.deepfold.org).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TOKEN") or os.environ.get("ADMIN_TOKEN"),
        help="bearer token. Optional -- /v2/align/pairwise is the public "
        "endpoint and does not require auth. Only pass a token when "
        "pointing at a gated deployment (default $TOKEN, else $ADMIN_TOKEN).",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="path to a FASTA file containing the query (one record). Use '-' for stdin.",
    )
    parser.add_argument(
        "--query-seq",
        type=str,
        help="inline query residues, used when --query is not supplied.",
    )
    parser.add_argument(
        "--query-id",
        type=str,
        default="query",
        help="query id stamped into stats and (optionally) the A3M output.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="path to a FASTA file containing one or more targets. Use '-' for stdin.",
    )
    parser.add_argument(
        "--target-seq",
        action="append",
        type=_parse_inline_target,
        default=[],
        metavar="ID:SEQ",
        help="inline target. Repeat for multiple targets. Mixable with --targets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="PLM backend (ankh_cl / ankh_large / esm1b / prott5). Default: server's.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="OTalign DP mode (local / global / glocal / q2t / t2q). Default: server's.",
    )
    parser.add_argument(
        "--aligner",
        type=str,
        default=None,
        help="aligner id. Default: server's (otalign).",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra aligner tunable forwarded in `options`. Repeatable. "
        'VALUE is JSON-decoded (so KEY=1.5 sends a float, KEY="x" sends a string).',
    )
    parser.add_argument(
        "--sort-by-score",
        action="store_true",
        help="emit alignments in score-descending order (best hit first). "
        "When --chunk-size splits the request, sorting is applied globally "
        "across the merged result.",
    )
    parser.add_argument(
        "--no-a3m",
        dest="emit_a3m",
        action="store_false",
        default=True,
        help="suppress the rendered A3M payload in the response.",
    )
    parser.add_argument(
        "--out-a3m",
        type=str,
        default=None,
        help="write the A3M payload here. Implied --emit-a3m.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="write the full JSON response here (alignments + stats + a3m).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="split targets into chunks of at most N records and post each chunk "
        "in a separate /v2/align/pairwise call. Useful when the target list "
        "exceeds the server's `PairwiseAlignConfig.max_records` (default 5000). "
        "Unset ->single request.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="N",
        help="drop targets whose residue length exceeds N before posting "
        "(the server rejects the whole request on the first over-long record). "
        "Default: no client-side filter. Set to 1022 to match the server's "
        "default per-record cap.",
    )
    parser.add_argument(
        "--out-hist",
        type=str,
        default=None,
        metavar="PATH",
        help="write a score histogram PNG to PATH. Lazy-imports matplotlib.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=80,
        help="histogram bin count (default 80).",
    )
    parser.add_argument(
        "--hist-title",
        type=str,
        default=None,
        help="histogram title override.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="HTTP timeout in seconds (default 900, per request).",
    )

    args = parser.parse_args(argv)

    # `--token` is optional: /v2/align/pairwise is the public endpoint.
    # No fail-fast here -- the request goes through unauthenticated when
    # no token is supplied.

    # Resolve query.
    if args.query and args.query_seq:
        parser.error("--query and --query-seq are mutually exclusive.")
    if args.query:
        records = read_fasta(args.query)
        if not records:
            parser.error(f"no records in {args.query}")
        if len(records) > 1:
            print(
                f"warning: {args.query} has {len(records)} records; "
                f"using the first ({records[0][0]!r}).",
                file=sys.stderr,
            )
        query_id = records[0][0] or args.query_id
        query_seq = records[0][1]
    elif args.query_seq:
        query_id = args.query_id
        query_seq = args.query_seq
    else:
        parser.error("need --query or --query-seq.")

    # Resolve targets.
    targets: list[dict[str, str]] = []
    if args.targets:
        for tid, seq in read_fasta(args.targets):
            targets.append({"id": tid, "sequence": seq})
    for tid, seq in args.target_seq:
        targets.append({"id": tid, "sequence": seq})
    if not targets:
        parser.error("need at least one target (--targets or --target-seq).")

    # Optional client-side length filter.
    skipped_too_long = 0
    if args.max_length is not None:
        kept: list[dict[str, str]] = []
        for t in targets:
            if len(t["sequence"]) > args.max_length:
                skipped_too_long += 1
            else:
                kept.append(t)
        if skipped_too_long:
            print(
                f"warning: dropped {skipped_too_long} target(s) "
                f"with len > {args.max_length}.",
                file=sys.stderr,
            )
        targets = kept
        if not targets:
            parser.error("all targets dropped by --max-length filter; nothing to send.")

    # Decode --option entries.
    options: dict[str, object] = {}
    for raw in args.option:
        if "=" not in raw:
            parser.error(f"--option entries must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            # Fall back to a plain string --convenient for free-text values.
            options[key] = value

    extra_body: dict[str, object] = {}
    if args.model is not None:
        extra_body["model"] = args.model
    if args.mode is not None:
        extra_body["mode"] = args.mode
    if args.aligner is not None:
        extra_body["aligner"] = args.aligner
    if options:
        extra_body["options"] = options

    emit_a3m = args.emit_a3m or bool(args.out_a3m)
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else len(targets)

    try:
        response = run_chunked(
            base_url=args.base_url,
            token=args.token,
            query_id=query_id,
            query_sequence=query_seq,
            targets=targets,
            chunk_size=chunk_size,
            timeout_s=args.timeout,
            emit_a3m=emit_a3m,
            sort_by_score=args.sort_by_score,
            extra_body=extra_body,
            progress=args.chunk_size is not None,
        )
    except ApiError as e:
        print(f"error: {e.code}: {e.message}", file=sys.stderr)
        if e.detail:
            print(json.dumps(e.detail, indent=2), file=sys.stderr)
        return 2

    # Pretty-print a summary table.
    stats = response.get("stats", {})
    alignments = response.get("alignments", [])
    print("=" * 72)
    print(f"query              {query_id}  (len={stats.get('query_length')})")
    print(f"model / mode       {stats.get('model')} / {stats.get('mode')}")
    print(
        f"targets in/kept    {stats.get('targets_in')} / {stats.get('targets_kept')}"
        f"  (dropped no-match: {stats.get('targets_dropped_no_match')})"
    )
    print(f"unique target seqs {stats.get('unique_target_seqs')}")
    if stats.get("chunks", 1) > 1:
        print(
            f"chunks             {stats.get('chunks')}"
            f"  (elapsed {stats.get('elapsed_s')}s)"
        )
    if skipped_too_long:
        print(f"client-side skipped {skipped_too_long} target(s) over --max-length")
    print("-" * 72)
    # Cap the per-target dump to keep the terminal usable on big runs.
    head = alignments[:50]
    print(f"{'target':<24s} {'score':>9s}  q[start:end)    t[start:end)")
    for a in head:
        print(
            f"{a['target_id']:<24.24s} "
            f"{a['score']:>9.3f}  "
            f"[{a['query_start']:>5d}:{a['query_end']:<5d})  "
            f"[{a['target_start']:>5d}:{a['target_end']:<5d})"
        )
    if len(alignments) > len(head):
        print(f"... ({len(alignments) - len(head)} more --full list in --out-json/--out-a3m)")

    # Optional dumps.
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(response, indent=2))
        print(f"\nwrote JSON response -> {args.out_json}")

    if args.out_a3m:
        a3m_text = response.get("a3m") or ""
        Path(args.out_a3m).write_text(a3m_text)
        print(f"wrote A3M payload -> {args.out_a3m}")

    if args.out_hist:
        scores = [float(a["score"]) for a in alignments if "score" in a]
        write_histogram(
            scores=scores,
            out_path=Path(args.out_hist),
            bins=args.hist_bins,
            title=args.hist_title,
        )
        print(
            f"wrote histogram -> {args.out_hist}"
            f"  (n={len(scores)}"
            + (
                f", range {min(scores):.3f}...{max(scores):.3f}"
                if scores
                else ""
            )
            + ")"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
