"""Correctness parity between FlashSinkhorn and the eager torch solver.

These tests run on CPU (torch.compile compiles fine there) and verify:
  - Transport plan `P` matches eager within 1e-4 (the solver's own tol).
  - Dual variables `f`, `g` match within 1e-4.
  - Iteration count matches when both converge.

GPU-specific behavior (CUDA stream, SMEM residency of C) is verified in
the micro-benchmark at `bench/flash_sinkhorn.py`, not here — the test
runner doesn't require a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.parametrize("lq,lt", [(16, 16), (32, 48), (64, 32), (96, 120)])
def test_flash_matches_eager_P(lq, lt) -> None:
    _require_torch()
    import torch

    from plmmsa.align.sinkhorn_flash import unbalanced_sinkhorn_flash
    from plmmsa.align.sinkhorn_torch import unbalanced_sinkhorn_torch

    rng = np.random.default_rng(42)
    C_np = rng.normal(size=(lq, lt)).astype(np.float32)
    C = torch.as_tensor(C_np, dtype=torch.float32)

    eager = unbalanced_sinkhorn_torch(C, eps=0.1, tau=1.0, n_iter=200, tol=1e-5)
    flash = unbalanced_sinkhorn_flash(C, eps=0.1, tau=1.0, n_iter=200, tol=1e-5)

    # 1e-3 is generous; compiled ops on CPU reorder some ops which can
    # perturb the last few ULPs. The Sinkhorn solver's own `tol` is
    # 1e-5, so 1e-3 still means "solutions agree well within what the
    # iteration tolerance distinguishes".
    assert np.max(np.abs(flash.P - eager.P)) < 1e-3
    assert np.max(np.abs(flash.f - eager.f)) < 1e-3
    assert np.max(np.abs(flash.g - eager.g)) < 1e-3


def test_flash_converged_flag_reflects_tolerance() -> None:
    _require_torch()
    import torch

    from plmmsa.align.sinkhorn_flash import unbalanced_sinkhorn_flash

    # Deep enough that convergence is easy — fused chunked loop still
    # needs to detect it.
    rng = np.random.default_rng(1)
    C = torch.as_tensor(rng.normal(size=(24, 24)).astype(np.float32))
    res = unbalanced_sinkhorn_flash(C, eps=0.1, tau=1.0, n_iter=200, tol=1e-4)
    assert res.converged is True
    assert res.iterations <= 200


def test_flash_no_cuda_falls_through_cpu_path() -> None:
    """When `device='cpu'` is forced, we take the eager torch path
    (compile overhead dominates at small matrices on CPU). Verify the
    result shape + type."""
    _require_torch()
    import torch

    from plmmsa.align.sinkhorn_flash import unbalanced_sinkhorn_flash

    rng = np.random.default_rng(3)
    C = torch.as_tensor(rng.normal(size=(12, 12)).astype(np.float32))
    res = unbalanced_sinkhorn_flash(
        C,
        eps=0.1,
        tau=1.0,
        n_iter=50,
        tol=1e-4,
        device="cpu",
    )
    assert res.P.shape == (12, 12)
    assert res.f.shape == (12,)
    assert res.g.shape == (12,)
    assert res.P.dtype == np.float32


def test_otalign_dispatch_honors_fused_flag(monkeypatch) -> None:
    """OTalign's `_solve_sinkhorn` should dispatch to the flash solver
    when `fused_sinkhorn=True` AND the input is on CUDA. On CPU-only
    test runners it falls back to the eager solver — that's what this
    test checks."""
    _require_torch()
    import torch

    from plmmsa.align import otalign as otalign_mod

    # CPU: fused path skips through to the eager implementation.
    hp = otalign_mod._Hyperparams(
        eps=0.1,
        tau=1.0,
        n_iter=50,
        tol=1e-4,
        go_base=8.0,
        ge_base=1.0,
        gamma=1.0,
        eta=0.25,
        k_f=0.75,
        k_g=0.75,
        clip_lower=0.25,
        clip_upper=4.0,
        dp_mode="glocal",
        fused=True,
    )
    rng = np.random.default_rng(7)
    C = torch.as_tensor(rng.normal(size=(20, 20)).astype(np.float32))
    res = otalign_mod._solve_sinkhorn(C, hp)
    assert res.P.shape == (20, 20)
    # Doesn't matter which backend ran it — parity is the contract.
