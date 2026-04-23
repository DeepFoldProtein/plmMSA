"""OTalign — optimal-transport-based pairwise aligner.

End-to-end flow for one (query, target) pair:

  1. Build cost matrix `C = 1 - cos(q, t)` (per-residue PLM embeddings).
  2. Solve unbalanced Sinkhorn with KL-relaxed marginals → plan `P` +
     dual potentials `f, g`.
  3. Derive per-cell match scores `S` from `P` via pointwise mutual
     information (PMI), and per-residue gap-cost multipliers from `f, g`
     + marginal mass of `P`.
  4. Run an affine-gap DP with per-residue gap costs (see `otalign_dp`).
  5. Report `Alignment` with `score = sum(P[qi, ti] for matched pairs)`
     — per the operator's definition ("sum of matched position of the
     result plan matrix"), *not* the DP's own PMI objective.

This is an embedding-level aligner — OT needs both residue sets directly,
can't operate on a precomputed similarity matrix, so `OTalign` inherits
from `Aligner`, not `MatrixAligner`.

Re-authored from the algorithmic spec (pLM-BLAST / OTalign paper +
DeepFoldProtein/OTalign as reference) — not a verbatim port.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from plmmsa.align.base import Aligner, Alignment, AlignMode
from plmmsa.align.otalign_dp import DPMode, affine_gap_dp
from plmmsa.align.sinkhorn import SinkhornResult, unbalanced_sinkhorn


class OTalign(Aligner):
    """Optimal-transport aligner with position-specific gap costs."""

    id = "otalign"
    display_name = "OTalign (unbalanced Sinkhorn + affine-gap DP)"

    # Sinkhorn defaults — operator-provided calibration.
    DEFAULT_EPS = 0.1
    DEFAULT_TAU = 1.0
    DEFAULT_N_ITER = 100
    DEFAULT_TOL = 1e-4

    # Gap-factor defaults — upstream conventions.
    DEFAULT_GO_BASE = 8.0
    DEFAULT_GE_BASE = 1.0
    DEFAULT_GAMMA = 1.0       # exponent on mass-normalized factor
    DEFAULT_ETA = 0.25        # lower bound on ge multiplier
    DEFAULT_K_F = 0.75        # sigmoid sharpness on query-side dual var
    DEFAULT_K_G = 0.75        # sigmoid sharpness on target-side dual var
    DEFAULT_CLIP_LOWER = 0.25
    DEFAULT_CLIP_UPPER = 4.0
    DEFAULT_DP_MODE: DPMode = "glocal"

    def align(
        self,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        *,
        mode: AlignMode = "local",
        # Sinkhorn overrides
        eps: float | None = None,
        tau: float | None = None,
        n_iter: int | None = None,
        tol: float | None = None,
        # Gap-factor overrides
        go_base: float | None = None,
        ge_base: float | None = None,
        gamma: float | None = None,
        eta: float | None = None,
        k_f: float | None = None,
        k_g: float | None = None,
        clip_lower: float | None = None,
        clip_upper: float | None = None,
        # DP mode — OTalign natively supports all five AlignMode values
        # (local, global, glocal, q2t, t2q). By default the DP runs with
        # whatever `mode` says — no silent substitutions. `dp_mode` is
        # kept as an explicit override escape hatch (back-compat + lets
        # callers pass a mode that differs from `mode` if they want).
        dp_mode: DPMode | None = None,
        # Torch device override. `None` → auto-detect (GPU if available).
        # "" / "cpu" → force CPU. "cuda:0" etc → pin to that device.
        device: str | None = None,
        **_: Any,
    ) -> list[Alignment]:
        settings = _Hyperparams(
            eps=_opt(eps, self.DEFAULT_EPS),
            tau=_opt(tau, self.DEFAULT_TAU),
            n_iter=_opt(n_iter, self.DEFAULT_N_ITER),
            tol=_opt(tol, self.DEFAULT_TOL),
            go_base=_opt(go_base, self.DEFAULT_GO_BASE),
            ge_base=_opt(ge_base, self.DEFAULT_GE_BASE),
            gamma=_opt(gamma, self.DEFAULT_GAMMA),
            eta=_opt(eta, self.DEFAULT_ETA),
            k_f=_opt(k_f, self.DEFAULT_K_F),
            k_g=_opt(k_g, self.DEFAULT_K_G),
            clip_lower=_opt(clip_lower, self.DEFAULT_CLIP_LOWER),
            clip_upper=_opt(clip_upper, self.DEFAULT_CLIP_UPPER),
            dp_mode=_opt(dp_mode, _mode_to_dp(mode)),
        )
        # Device resolution: explicit kwarg > auto-detect. When auto,
        # pick `cuda:0` if torch.cuda.is_available, else stay CPU. This
        # means a GPU-enabled align container transparently uses it.
        resolved_device = _resolve_device(device)

        # Coerce inputs to the resolved backend so the whole pipeline
        # (cost matrix, Sinkhorn, PMI, gap factors) runs on-device.
        q = _l2_normalize_any(_to_backend(query_embedding, resolved_device))
        out: list[Alignment] = []
        for t_emb in target_embeddings:
            t = _l2_normalize_any(_to_backend(t_emb, resolved_device))
            out.append(_align_pair(q, t, settings))
        return out


# --- internals ---------------------------------------------------------------


class _Hyperparams:
    """Resolved hyperparameters for one align() call. Not a dataclass so
    we can stack defaults + overrides without verbose constructors."""

    __slots__ = (
        "clip_lower",
        "clip_upper",
        "dp_mode",
        "eps",
        "eta",
        "gamma",
        "ge_base",
        "go_base",
        "k_f",
        "k_g",
        "n_iter",
        "tau",
        "tol",
    )

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _align_pair(q, t, hp: _Hyperparams) -> Alignment:
    """One query/target pair → one Alignment.

    Glue in four steps — each step is a small algorithmic unit with its
    own docstring upstream. The only "new" math here is the PMI score +
    gap-factor derivation (§3 of the module docstring).

    Input tensors may be numpy arrays or torch tensors. Torch tensors
    flow through the torch Sinkhorn on their own device (CPU or any
    CUDA); numpy inputs stay on CPU via the pure-numpy Sinkhorn. Either
    way the DP runs on CPU numpy because the per-cell affine-gap loop
    is inherently sequential.
    """
    # Step 1 — cost matrix, bounded [0, 2]. Done in the native backend of
    # the inputs so we don't leave the GPU until the DP step below.
    C = _cost_matrix(q, t)

    # Step 2 — unbalanced Sinkhorn with KL-relaxed marginals. Dispatch on
    # C's backend: torch tensor → torch solver on C.device; numpy → CPU.
    result = _solve_sinkhorn(C, hp)
    P = result.P

    # Step 3 — PMI match scores + per-residue gap factors.
    S = _pmi_from_plan(P)
    go_q, ge_q, go_t, ge_t = _gap_factors(
        P, result.f, result.g,
        go_base=hp.go_base, ge_base=hp.ge_base,
        gamma=hp.gamma, eta=hp.eta,
        k_f=hp.k_f, k_g=hp.k_g,
        clip_lower=hp.clip_lower, clip_upper=hp.clip_upper,
    )

    # Step 4 — DP.
    dp = affine_gap_dp(
        S, go_q=go_q, ge_q=ge_q, go_t=go_t, ge_t=ge_t, mode=hp.dp_mode,
    )

    # Step 5 — headline score = sum of transport-plan mass at matched
    # cells. The DP's own PMI sum is stored as a diagnostic if callers
    # want it (accessible via DPResult but we don't surface it here to
    # keep the Alignment contract stable).
    matched = [(qi, ti) for qi, ti in dp.columns if qi >= 0 and ti >= 0]
    if matched:
        rows = np.asarray([qi for qi, _ in matched], dtype=np.int64)
        cols = np.asarray([ti for _, ti in matched], dtype=np.int64)
        plan_sum = float(P[rows, cols].sum())
    else:
        plan_sum = 0.0

    return Alignment(
        score=plan_sum,
        mode=_aligner_mode(hp.dp_mode),
        query_start=dp.q_start,
        query_end=dp.q_end,
        target_start=dp.t_start,
        target_end=dp.t_end,
        columns=list(dp.columns),
    )


def _pmi_from_plan(P: np.ndarray, epsilon: float = 1e-30) -> np.ndarray:
    """Pointwise mutual information of the transport plan.

    `S[i, j] = log p(i, j) - log p(i) - log p(j)` where `p(i, j)` is the
    plan `P` renormalized to a joint distribution. A small `epsilon`
    floor keeps `log(0)` finite — relevant for cells where Sinkhorn
    returned ~0 mass. Those cells get a very negative match score, which
    the DP correctly treats as "don't match here."
    """
    total = float(P.sum())
    if total <= 0.0:
        return np.zeros_like(P, dtype=np.float32)
    pij = P / total
    pi = pij.sum(axis=1, keepdims=True)  # (Lq, 1)
    pj = pij.sum(axis=0, keepdims=True)  # (1, Lt)
    return (
        np.log(np.clip(pij, epsilon, None))
        - np.log(np.clip(pi, epsilon, None))
        - np.log(np.clip(pj, epsilon, None))
    ).astype(np.float32)


def _gap_factors(
    P: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    *,
    go_base: float,
    ge_base: float,
    gamma: float,
    eta: float,
    k_f: float,
    k_g: float,
    clip_lower: float,
    clip_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Position-specific gap-open / gap-extend vectors.

    Combines two signals into a `(0, 1]`-ish multiplier per residue:
      (a) **mass ratio** — how much of the plan's mass landed on this
          residue, relative to the median row/col mass. A cell with lots
          of inbound mass is "well-matched"; few is "poorly matched."
      (b) **dual-variable z-score through a sigmoid** — captures how
          hard Sinkhorn had to push to satisfy the marginal at this
          residue. Converted to `(0, 1)` via
          `sigmoid(-k * normalized_dual)` so residues with high dual
          values get smaller multipliers (cheap matches → no gaps here).

    Multiplying (a) by (b) gives one factor per residue; final gap_open /
    gap_extend is `max(base_ext, base * factor)` so the extend never
    goes above open.
    """
    row_mass = P.sum(axis=1)  # (Lq,)
    col_mass = P.sum(axis=0)  # (Lt,)

    go_q, ge_q = _gap_factors_one_side(
        mass=row_mass, dual=f,
        go_base=go_base, ge_base=ge_base,
        gamma=gamma, eta=eta, k=k_f,
        clip_lower=clip_lower, clip_upper=clip_upper,
    )
    go_t, ge_t = _gap_factors_one_side(
        mass=col_mass, dual=g,
        go_base=go_base, ge_base=ge_base,
        gamma=gamma, eta=eta, k=k_g,
        clip_lower=clip_lower, clip_upper=clip_upper,
    )
    return go_q, ge_q, go_t, ge_t


def _gap_factors_one_side(
    *,
    mass: np.ndarray,
    dual: np.ndarray,
    go_base: float,
    ge_base: float,
    gamma: float,
    eta: float,
    k: float,
    clip_lower: float,
    clip_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(go, ge)` vectors for one side of the alignment."""
    # Median over nonzero mass (exactly-zero rows often mean masked
    # positions in upstream — skip them so they don't pull the median).
    nonzero = mass[mass > 0]
    med_mass = float(np.median(nonzero)) if nonzero.size else 1.0
    if med_mass <= 0.0:
        med_mass = 1.0
    ratio = np.clip(mass / med_mass, clip_lower, clip_upper).astype(np.float32)
    if gamma != 1.0:
        ratio = np.power(ratio, gamma, dtype=np.float32)

    dual = np.asarray(dual, dtype=np.float32)
    med_dual = float(np.median(dual))
    std_dual = float(dual.std())
    norm_dual = (dual - med_dual) / (std_dual + 1e-6)
    # sigmoid(-k * z) ∈ (0, 1). Negative exponent flips direction so
    # *low* dual values (hard-to-match residues) produce *larger*
    # factors → more expensive to gap here, which matches upstream.
    #
    # Implementation note: we clip the exponent to avoid overflow on
    # extreme z-scores when `k` is large or the dual distribution has
    # outliers.
    exponent = np.clip(-k * norm_dual, -60.0, 60.0)
    sigmoid = 1.0 / (1.0 + np.exp(-exponent))

    factor = (ratio * sigmoid).astype(np.float32)

    go = np.maximum(float(ge_base), float(go_base) * factor).astype(np.float32)
    ge = np.maximum(float(eta) * float(ge_base), float(ge_base) * factor).astype(np.float32)
    return go, ge


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _keep_tensor_or_numpy(x):
    """Pass torch tensors through untouched; convert everything else to
    a float32 numpy array. Used at the align() entry point so the GPU
    path survives all the way to Sinkhorn."""
    if _is_torch_tensor(x):
        return x
    return np.asarray(x, dtype=np.float32)


def _resolve_device(explicit):
    """Pick a torch device for OTalign's Sinkhorn + cost matrix.

    Precedence (highest first):
      1. `explicit` kwarg from the request (non-None, non-empty).
      2. `settings.aligners.otalign.device` when it resolves to a real
         torch device id — passed here via the explicit kwarg by the
         align service.
      3. The PLM device pin for OTalign's default score_model
         (Ankh-Large). Read from `ANKH_LARGE_DEVICE` if present — same
         env var the embedding container uses to place Ankh-Large, so
         the two stay in sync automatically.
      4. `cuda:0` if torch.cuda.is_available().
      5. `cpu`.
    """
    if explicit is not None:
        s = explicit.strip()
        if s:
            return s

    # Match the embedding service's Ankh-Large pin (OTalign's default
    # score_model). Keeps the tensor on the same GPU Ankh-Large used
    # to produce it — no P2P copy.
    pinned = os.environ.get("ANKH_LARGE_DEVICE", "").strip()
    if pinned:
        return pinned

    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_backend(x, device: str):
    """Cast the input into the requested backend.

    - device == "cpu": stay numpy.
    - device starts with "cuda" / other torch device id: promote to
      a torch tensor on that device.
    - input already a torch tensor: `.to(device)` to move if needed.
    """
    if device == "cpu" and not _is_torch_tensor(x):
        return np.asarray(x, dtype=np.float32)

    import torch

    target = torch.device(device)
    if _is_torch_tensor(x):
        return x.to(device=target, dtype=torch.float32)
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return torch.as_tensor(arr, dtype=torch.float32, device=target)


def _is_torch_tensor(x) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _l2_normalize_any(x):
    """Per-row L2 normalize on either torch or numpy input."""
    if _is_torch_tensor(x):
        import torch

        norms = torch.linalg.norm(x, dim=-1, keepdim=True)
        return x / torch.clamp(norms, min=1e-12)
    return _l2_normalize(x)


def _cost_matrix(q, t):
    """`1 - cosine_similarity(q, t)`. Stays on the same backend as the inputs."""
    if _is_torch_tensor(q) or _is_torch_tensor(t):
        import torch

        # Bring both to the same device / dtype. If q is torch and t is
        # numpy (or vice versa), promote t to torch on q's device.
        if not _is_torch_tensor(q):
            q = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=t.device)
        if not _is_torch_tensor(t):
            t = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=q.device)
        return 1.0 - (q @ t.T)
    return 1.0 - (q @ t.T)


def _solve_sinkhorn(C, hp: _Hyperparams) -> SinkhornResult:
    """Dispatch to the torch solver when C is a tensor, else numpy."""
    if _is_torch_tensor(C):
        from plmmsa.align.sinkhorn_torch import unbalanced_sinkhorn_torch

        return unbalanced_sinkhorn_torch(
            C, eps=hp.eps, tau=hp.tau, n_iter=hp.n_iter, tol=hp.tol,
        )
    return unbalanced_sinkhorn(
        C, eps=hp.eps, tau=hp.tau, n_iter=hp.n_iter, tol=hp.tol,
    )


def _opt(value: Any, default: Any) -> Any:
    return default if value is None else value


_VALID_DP_MODES: frozenset[str] = frozenset(
    ("local", "global", "glocal", "q2t", "t2q"),
)


def _mode_to_dp(mode: AlignMode) -> DPMode:
    """Pass the AlignMode straight through as the OTalign DPMode.

    All five AlignMode values are valid DPMode values, so the mapping
    is the identity. Kept as a named function (rather than inlined) so
    the callsite documents that no substitution happens — earlier
    drafts of this module silently mapped `global` onto `glocal`, which
    produced alignments that disagreed with what the caller asked for.
    """
    if mode not in _VALID_DP_MODES:
        raise ValueError(
            f"OTalign: unsupported mode {mode!r}; expected one of "
            f"{sorted(_VALID_DP_MODES)}"
        )
    return mode  # type: ignore[return-value]


def _aligner_mode(dp_mode: DPMode) -> AlignMode:
    """Reverse mapping so the returned `Alignment.mode` matches the
    aligner-base enum."""
    return "local" if dp_mode == "local" else "global"


__all__ = ["OTalign"]
