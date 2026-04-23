"""Log-domain unbalanced Sinkhorn solver for OTalign.

Re-authored from first principles (not a port of the upstream code).
The public surface:

    P, f, g = unbalanced_sinkhorn(C, a=None, b=None, eps=0.1, tau=1.0,
                                  n_iter=100, tol=1e-4)

Returns the transport plan `P[i, j]` along with the dual potentials
`f[i]` and `g[j]` that downstream consumers use to compute position-
specific gap penalties.

Math
----
Balanced Sinkhorn iterates dual potentials `f, g` against
`a ∈ R^Lq, b ∈ R^Lt` marginals. Unbalanced Sinkhorn with KL-relaxed
marginals adds a `τ`-scaled penalty that softens the constraint, and the
log-domain update becomes:

    f[i] ← (τ / (τ + ε)) · (ε · log a[i] - ε · lse_j( (g[j] - C[i, j]) / ε ))
    g[j] ← (τ / (τ + ε)) · (ε · log b[j] - ε · lse_i( (f[i] - C[i, j]) / ε ))

where `lse` is `log-sum-exp`. The plan at the end is:

    P[i, j] = exp( (f[i] + g[j] - C[i, j]) / ε )

Shape / dtype
-------------
`C` is `(Lq, Lt)` float32. Output `P` is float32 of the same shape. `a`
and `b` default to uniform 1/Lq, 1/Lt — that's what upstream OTalign uses
across global / glocal / local modes.

Calibration defaults (`eps=0.1, tau=1.0, n_iter=100`) come from the
operator's guidance and match upstream behavior within numerical noise;
convergence is typically reached by iteration ~80 on `0 ≤ C ≤ 2` cost
matrices produced by cosine-distance from per-residue PLM embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SinkhornResult:
    """What the solver returns + convergence diagnostics.

    `iterations` reflects the number of dual updates actually performed
    (< `n_iter` when `tol` is reached early). `converged` is `False` if
    the `n_iter` budget was exhausted without crossing `tol`; the plan
    is still usable — the dual variables just haven't fully stabilized.
    """

    P: np.ndarray  # (Lq, Lt) — transport plan
    f: np.ndarray  # (Lq,)   — query-side dual potential
    g: np.ndarray  # (Lt,)   — target-side dual potential
    iterations: int
    converged: bool


def unbalanced_sinkhorn(
    C: np.ndarray,
    *,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    eps: float = 0.1,
    tau: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-4,
) -> SinkhornResult:
    """Solve unbalanced OT with KL-relaxed marginals, log-domain.

    Parameters
    ----------
    C : (Lq, Lt) float array
        Cost matrix. Typically `1 - cos(q, t)` so values live in `[0, 2]`.
    a : (Lq,) float array, optional
        Source marginal. Default: uniform `1/Lq`.
    b : (Lt,) float array, optional
        Target marginal. Default: uniform `1/Lt`.
    eps : float
        Entropic regularizer. Smaller = sharper plan, slower convergence
        and more numerical instability. `0.1` is the OTalign default.
    tau : float
        Marginal-relaxation strength. As `tau → ∞` the solver becomes
        balanced Sinkhorn. `1.0` lets the plan deviate gracefully from
        the marginals when one side has mass the other can't absorb.
    n_iter : int
        Maximum Gauss-Seidel updates.
    tol : float
        Max-abs change in `(f, g)` to call convergence.

    Returns
    -------
    SinkhornResult
        `(P, f, g, iterations, converged)` — see dataclass.
    """
    C = np.ascontiguousarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"C must be 2-D, got shape {C.shape}")
    lq, lt = C.shape

    a_arr: np.ndarray = (
        np.full(lq, 1.0 / lq, dtype=np.float32)
        if a is None
        else np.ascontiguousarray(a, dtype=np.float32)
    )
    b_arr: np.ndarray = (
        np.full(lt, 1.0 / lt, dtype=np.float32)
        if b is None
        else np.ascontiguousarray(b, dtype=np.float32)
    )
    if a_arr.shape != (lq,):
        raise ValueError(f"a shape {a_arr.shape} does not match Lq={lq}")
    if b_arr.shape != (lt,):
        raise ValueError(f"b shape {b_arr.shape} does not match Lt={lt}")

    # Log-space priors. Guard log(0) by clamping to a small positive
    # value — the caller shouldn't pass exact zeros anyway but mask
    # regions in upstream OT pipelines do.
    log_a = np.log(np.clip(a_arr, 1e-30, None))
    log_b = np.log(np.clip(b_arr, 1e-30, None))

    scale = tau / (tau + eps)

    # Initialize dual potentials at zero (u = v = 1 in primal space).
    f = np.zeros(lq, dtype=np.float32)
    g = np.zeros(lt, dtype=np.float32)

    converged = False
    iters = 0
    for step in range(1, n_iter + 1):
        iters = step
        f_prev = f
        g_prev = g

        # lse over j for each i: log-sum-exp across target residues.
        # Subtract the per-row max for numerical stability.
        m_j = (g[None, :] - C) / eps  # (Lq, Lt)
        m_max = m_j.max(axis=1, keepdims=True)
        lse_q = (m_max.squeeze(1) + np.log(
            np.exp(m_j - m_max).sum(axis=1)
        )).astype(np.float32)
        f = (scale * (eps * log_a - eps * lse_q)).astype(np.float32)

        m_i = (f[:, None] - C) / eps  # (Lq, Lt)
        m_max_i = m_i.max(axis=0, keepdims=True)
        lse_t = (m_max_i.squeeze(0) + np.log(
            np.exp(m_i - m_max_i).sum(axis=0)
        )).astype(np.float32)
        g = (scale * (eps * log_b - eps * lse_t)).astype(np.float32)

        df = float(np.max(np.abs(f - f_prev)))
        dg = float(np.max(np.abs(g - g_prev)))
        if max(df, dg) < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "sinkhorn: did not converge in %d iters (eps=%g, tau=%g)",
            n_iter, eps, tau,
        )

    # Primal plan from dual potentials.
    P = np.exp((f[:, None] + g[None, :] - C) / eps).astype(np.float32)

    return SinkhornResult(P=P, f=f, g=g, iterations=iters, converged=converged)


__all__ = ["SinkhornResult", "unbalanced_sinkhorn"]
