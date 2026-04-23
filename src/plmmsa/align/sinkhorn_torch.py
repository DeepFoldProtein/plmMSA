"""Torch variant of the unbalanced-Sinkhorn solver.

Same math as `plmmsa.align.sinkhorn.unbalanced_sinkhorn` (log-domain
UOT with KL-relaxed marginals) — re-expressed with torch ops so it
runs on whatever device the input tensors live on. No hardcoded GPU:
- torch-tensor input → solver runs on `C.device`.
- numpy input → caller chooses (default CPU).

Output `SinkhornResult.P / f / g` are numpy arrays to match the CPU
solver's contract; we move to CPU at the end so downstream code
(DP, numpy-based gap-factor derivation) stays backend-agnostic.

Kept in a separate module from `sinkhorn.py` so deployments without
torch (if we ever strip it from the align image) still boot — the
pure-numpy reference is always available.
"""

from __future__ import annotations

import logging

import numpy as np

from plmmsa.align.sinkhorn import SinkhornResult

logger = logging.getLogger(__name__)


def unbalanced_sinkhorn_torch(
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
    """Log-domain unbalanced Sinkhorn on torch.

    Device selection (highest precedence first):
      1. Explicit `device=` kwarg.
      2. `C.device` if `C` is already a torch tensor.
      3. CPU.
    """
    import torch

    # Resolve target device.
    if device is not None:
        dev = torch.device(device)
    elif isinstance(C, torch.Tensor):
        dev = C.device
    else:
        dev = torch.device("cpu")

    C_t = _to_tensor(C, device=dev)
    if C_t.ndim != 2:
        raise ValueError(f"C must be 2-D, got shape {tuple(C_t.shape)}")
    lq, lt = C_t.shape

    a_t = (
        torch.full((lq,), 1.0 / lq, dtype=torch.float32, device=dev)
        if a is None
        else _to_tensor(a, device=dev)
    )
    b_t = (
        torch.full((lt,), 1.0 / lt, dtype=torch.float32, device=dev)
        if b is None
        else _to_tensor(b, device=dev)
    )
    if a_t.shape != (lq,):
        raise ValueError(f"a shape {tuple(a_t.shape)} does not match Lq={lq}")
    if b_t.shape != (lt,):
        raise ValueError(f"b shape {tuple(b_t.shape)} does not match Lt={lt}")

    log_a = torch.log(torch.clamp(a_t, min=1e-30))
    log_b = torch.log(torch.clamp(b_t, min=1e-30))

    scale = tau / (tau + eps)

    f = torch.zeros(lq, dtype=torch.float32, device=dev)
    g = torch.zeros(lt, dtype=torch.float32, device=dev)

    converged = False
    iters = 0
    for step in range(1, n_iter + 1):
        iters = step
        f_prev = f
        g_prev = g

        # Gauss-Seidel updates: alternate f, g with log-sum-exp across
        # the other axis.
        m_j = (g.unsqueeze(0) - C_t) / eps  # (Lq, Lt)
        lse_q = torch.logsumexp(m_j, dim=1)
        f = scale * (eps * log_a - eps * lse_q)

        m_i = (f.unsqueeze(1) - C_t) / eps  # (Lq, Lt)
        lse_t = torch.logsumexp(m_i, dim=0)
        g = scale * (eps * log_b - eps * lse_t)

        df = (f - f_prev).abs().max().item()
        dg = (g - g_prev).abs().max().item()
        if max(df, dg) < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "sinkhorn_torch: did not converge in %d iters (eps=%g, tau=%g, device=%s)",
            n_iter,
            eps,
            tau,
            dev,
        )

    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C_t) / eps)

    # Return numpy so downstream stays backend-agnostic.
    return SinkhornResult(
        P=P.detach().cpu().numpy().astype(np.float32),
        f=f.detach().cpu().numpy().astype(np.float32),
        g=g.detach().cpu().numpy().astype(np.float32),
        iterations=iters,
        converged=converged,
    )


def _to_tensor(obj, *, device):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(obj), dtype=torch.float32, device=device)


__all__ = ["unbalanced_sinkhorn_torch"]
