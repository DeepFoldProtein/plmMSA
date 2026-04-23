"""Micro-benchmark FlashSinkhorn vs. eager torch Sinkhorn.

Times the Sinkhorn step in isolation across matrix sizes that span the
CASP15 Lq x Lt distribution (Lq, Lt in {100, 200, 400, 800}) on the
GPU declared by `ANKH_LARGE_DEVICE` (fallback `cuda:0`). Prints a
wall-time table + a correctness check (max abs difference of P vs.
eager, must be < 1e-4).

Usage::

    uv run python bench/flash_sinkhorn.py
    uv run python bench/flash_sinkhorn.py --repeat 10 --sizes 100,200,400
    uv run python bench/flash_sinkhorn.py --device cpu  # sanity check

GPU-only by design — CPU flash path falls back to eager, so there's
nothing to measure. The CPU mode is only useful for debugging the
compile step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import mean, stdev

import numpy as np


def _parse_sizes(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FlashSinkhorn micro-benchmark")
    ap.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: env ANKH_LARGE_DEVICE → cuda:0 → cpu.",
    )
    ap.add_argument(
        "--sizes",
        default="100,200,400,800",
        help="Comma-separated Lq values; Lt uses the same list (Cartesian product).",
    )
    ap.add_argument(
        "--repeat", type=int, default=5, help="Inner-loop reps per (Lq, Lt) configuration."
    )
    ap.add_argument("--n-iter", type=int, default=100, help="Sinkhorn iteration budget.")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def _resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("ANKH_LARGE_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def _timed(fn, *, repeat: int) -> list[float]:
    """Run fn `repeat` times, return per-call wall times in seconds.

    For GPU timings we synchronize before and after each call so we
    don't measure just the dispatch. The first call warms up the cache;
    we drop it from the median-style summary.
    """
    try:
        import torch

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except ImportError:

        def _sync():
            pass

    out: list[float] = []
    _sync()
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append(time.perf_counter() - t0)
    return out


def main() -> int:
    args = build_argparser().parse_args()

    try:
        import torch
    except ImportError:
        print("torch not installed; install the `align` extra.", file=sys.stderr)
        return 2

    from plmmsa.align.sinkhorn_flash import unbalanced_sinkhorn_flash
    from plmmsa.align.sinkhorn_torch import unbalanced_sinkhorn_torch

    device = _resolve_device(args.device)
    sizes = _parse_sizes(args.sizes)
    rng = np.random.default_rng(args.seed)

    print(f"device: {device}  n_iter: {args.n_iter}  eps: {args.eps}  tol: {args.tol}")
    print()
    header = f"{'Lq':>5} {'Lt':>5} {'eager':>14} {'flash':>14} {'speedup':>8} {'max|ΔP|':>9}"
    print(header)
    print("-" * len(header))

    for lq in sizes:
        for lt in sizes:
            C_np = rng.normal(size=(lq, lt)).astype(np.float32)
            C = torch.as_tensor(C_np, dtype=torch.float32, device=device)

            # Warmup: first calls pay import / compile cost.
            _ = unbalanced_sinkhorn_torch(
                C,
                eps=args.eps,
                tau=args.tau,
                n_iter=args.n_iter,
                tol=args.tol,
            )
            _ = unbalanced_sinkhorn_flash(
                C,
                eps=args.eps,
                tau=args.tau,
                n_iter=args.n_iter,
                tol=args.tol,
            )

            # Bind loop vars via default args so Ruff B023 is satisfied and
            # the lambdas close over the right C / n_iter for this iteration.
            eager_times = _timed(
                lambda _C=C, _n=args.n_iter: unbalanced_sinkhorn_torch(
                    _C,
                    eps=args.eps,
                    tau=args.tau,
                    n_iter=_n,
                    tol=args.tol,
                ),
                repeat=args.repeat,
            )
            flash_times = _timed(
                lambda _C=C, _n=args.n_iter: unbalanced_sinkhorn_flash(
                    _C,
                    eps=args.eps,
                    tau=args.tau,
                    n_iter=_n,
                    tol=args.tol,
                ),
                repeat=args.repeat,
            )

            eager_mean = mean(eager_times) * 1000
            flash_mean = mean(flash_times) * 1000
            speedup = eager_mean / flash_mean if flash_mean > 0 else float("inf")

            # Correctness — one fresh call each, compare P.
            eager_res = unbalanced_sinkhorn_torch(
                C,
                eps=args.eps,
                tau=args.tau,
                n_iter=args.n_iter,
                tol=args.tol,
            )
            flash_res = unbalanced_sinkhorn_flash(
                C,
                eps=args.eps,
                tau=args.tau,
                n_iter=args.n_iter,
                tol=args.tol,
            )
            max_diff = float(np.max(np.abs(flash_res.P - eager_res.P)))

            print(
                f"{lq:>5} {lt:>5} {eager_mean:>11.2f} ms {flash_mean:>11.2f} ms "
                f"{speedup:>7.2f}x {max_diff:>9.2e}"
            )

    if args.repeat >= 2:
        print()
        print("(eager / flash = mean wall time over --repeat calls, post-warmup)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


_ = stdev  # re-export silence
