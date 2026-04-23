"""Fused-kernel unbalanced Sinkhorn — "FlashSinkhorn".

Same math + contract as `unbalanced_sinkhorn_torch` (log-domain UOT with
KL-relaxed marginals), but the iteration loop is wrapped in a single
`torch.compile` graph so the ~100 iterations don't each pay Python +
kernel-launch overhead.

This is the MVP. Expected win on the CASP15 OTalign workload:
~40-60 % of Phase 5 time is per-iteration kernel-launch latency at our
matrix sizes (Lq ≈ 100-300, Lt ≈ 100-400). A full Triton kernel that
also keeps `C` in SRAM across iterations would win more, and is the
next step once this wrapper is validated.

Design decisions:

- **fp32 throughout**, same as the eager path. Log-space residual
  subtractions accumulate precision loss at bf16/fp16 over ~100
  iterations — not worth the speedup.
- **Dynamic shapes**. The compiled inner function takes `Lq, Lt` as
  runtime args so we don't recompile per input tensor size (Lq, Lt
  vary per CASP15 target). Block size / unroll factors are
  constants inside the graph.
- **Autotune key on hidden dim** — not applicable to Sinkhorn itself
  (it operates on `C: (Lq, Lt)` after the matmul). The autotune hook
  belongs to the cost-matrix builder upstream; this module just
  consumes whatever `C` comes in. Kept here as a comment so future
  readers don't re-derive the decision.
- **Early stop is disabled inside the compiled loop**. `torch.compile`
  can't graph-break on `.item()` cheaply, and the eager path's
  typical convergence at ~30-50 iterations means we'd break early on
  most targets. Instead: run the full n_iter, but split into short
  chunks (`CHUNK_ITER` = 16) so we can still early-stop at chunk
  boundaries. Trade-off: up to CHUNK_ITER - 1 extra iterations per
  target when the eager solver would have stopped sooner. Cheap vs.
  the ~3x compiled speedup.
- **Fallback**: when `torch.compile` isn't available (old torch, no
  CUDA), return whatever `unbalanced_sinkhorn_torch` produces so
  callers don't have to special-case.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from plmmsa.align.sinkhorn import SinkhornResult
from plmmsa.align.sinkhorn_torch import unbalanced_sinkhorn_torch

logger = logging.getLogger(__name__)

# How many iterations run inside one compiled graph segment before we
# check convergence. Smaller → more frequent Python returns (slower but
# closer to the eager early-stop point). Larger → less overhead but
# more wasted iters past convergence. 16 is a reasonable trade at the
# n_iter=100 default.
CHUNK_ITER = 16


@lru_cache(maxsize=8)
def _compiled_chunk() -> Any:
    """Lazily compile the inner iteration chunk. Cached by the function
    identity so a single compile is shared across all FlashSinkhorn
    calls in a process.

    The returned callable has signature
        (C, log_a, log_b, f, g, eps, scale, n_steps) -> (f, g)
    with all tensors living on `C.device`.
    """
    import torch

    def _chunk(
        C: torch.Tensor,
        log_a: torch.Tensor,
        log_b: torch.Tensor,
        f: torch.Tensor,
        g: torch.Tensor,
        eps: float,
        scale: float,
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gauss-Seidel log-domain updates. Matches sinkhorn_torch.py
        # body line-for-line; `torch.compile` folds the loop into one
        # graph so the n_steps iterations don't pay per-step launch
        # overhead.
        for _ in range(n_steps):
            m_j = (g.unsqueeze(0) - C) / eps
            lse_q = torch.logsumexp(m_j, dim=1)
            f = scale * (eps * log_a - eps * lse_q)

            m_i = (f.unsqueeze(1) - C) / eps
            lse_t = torch.logsumexp(m_i, dim=0)
            g = scale * (eps * log_b - eps * lse_t)
        return f, g

    try:
        return torch.compile(_chunk, dynamic=True, fullgraph=True)
    except Exception:
        logger.exception("flash_sinkhorn: torch.compile failed; using eager loop")
        return _chunk


def unbalanced_sinkhorn_flash(
    C,  # np.ndarray or torch.Tensor
    *,
    a=None,
    b=None,
    eps: float = 0.1,
    tau: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-4,
    device: str | None = None,
) -> SinkhornResult:
    """Log-domain unbalanced Sinkhorn with compiled inner loop.

    Drop-in replacement for `unbalanced_sinkhorn_torch` — same args,
    same `SinkhornResult` return shape. Callers opt in via
    `aligners.otalign.fused_sinkhorn = true` in settings; the eager
    path stays the default until correctness + speedup are verified
    against the CASP15 baseline.
    """
    try:
        import torch
    except ImportError:
        # No torch → pure-numpy fallback. `sinkhorn.py`'s solver has
        # the same contract.
        from plmmsa.align.sinkhorn import unbalanced_sinkhorn

        return unbalanced_sinkhorn(
            np.ascontiguousarray(C, dtype=np.float32),
            a=None if a is None else np.ascontiguousarray(a, dtype=np.float32),
            b=None if b is None else np.ascontiguousarray(b, dtype=np.float32),
            eps=eps,
            tau=tau,
            n_iter=n_iter,
            tol=tol,
        )

    # Resolve device, matching sinkhorn_torch.
    if device is not None:
        dev = torch.device(device)
    elif isinstance(C, torch.Tensor):
        dev = C.device
    else:
        dev = torch.device("cpu")

    # torch.compile + CPU is possible but gives much smaller wins and
    # sometimes regresses on small matrices. Eager path there.
    if dev.type == "cpu":
        return unbalanced_sinkhorn_torch(
            C,
            a=a,
            b=b,
            eps=eps,
            tau=tau,
            n_iter=n_iter,
            tol=tol,
            device=device,
        )

    C_t = _as_tensor(C, dev=dev)
    if C_t.ndim != 2:
        raise ValueError(f"C must be 2-D, got shape {tuple(C_t.shape)}")
    lq, lt = C_t.shape

    a_t = (
        torch.full((lq,), 1.0 / lq, dtype=torch.float32, device=dev)
        if a is None
        else _as_tensor(a, dev=dev)
    )
    b_t = (
        torch.full((lt,), 1.0 / lt, dtype=torch.float32, device=dev)
        if b is None
        else _as_tensor(b, dev=dev)
    )
    log_a = torch.log(torch.clamp(a_t, min=1e-30))
    log_b = torch.log(torch.clamp(b_t, min=1e-30))
    scale = float(tau / (tau + eps))

    f = torch.zeros(lq, dtype=torch.float32, device=dev)
    g = torch.zeros(lt, dtype=torch.float32, device=dev)

    chunk = _compiled_chunk()
    iters = 0
    converged = False
    # Run full n_iter in CHUNK_ITER-sized compiled segments; check
    # convergence at each segment boundary.
    remaining = n_iter
    while remaining > 0:
        step = min(CHUNK_ITER, remaining)
        f_prev = f
        g_prev = g
        f, g = chunk(C_t, log_a, log_b, f, g, eps, scale, step)
        iters += step
        remaining -= step
        df = (f - f_prev).abs().max().item()
        dg = (g - g_prev).abs().max().item()
        if max(df, dg) < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "flash_sinkhorn: did not converge in %d iters (eps=%g, tau=%g, device=%s)",
            n_iter,
            eps,
            tau,
            dev,
        )

    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C_t) / eps)
    return SinkhornResult(
        P=P.detach().cpu().numpy().astype(np.float32),
        f=f.detach().cpu().numpy().astype(np.float32),
        g=g.detach().cpu().numpy().astype(np.float32),
        iterations=iters,
        converged=converged,
    )


def _as_tensor(obj, *, dev):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device=dev, dtype=torch.float32)
    return torch.as_tensor(np.asarray(obj), dtype=torch.float32, device=dev)


__all__ = ["CHUNK_ITER", "unbalanced_sinkhorn_flash"]
