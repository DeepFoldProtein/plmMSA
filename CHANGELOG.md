# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Upstream PLMAlign Algorithm 1 step 5 score-threshold filter. Default
  threshold `min(0.2·len(Q), 8.0)`. Per-aligner toggle via
  `[aligners.*].filter_enabled`, per-request override via
  `filter_by_score` on `/v2/msa`. `result.stats` now carries
  `hits_pre_filter`, `hits_post_filter`, `filter_applied`,
  `filter_threshold`.
- `torch` added to the align container image (new `align` extra) so
  OTalign's Sinkhorn + cost matrix can run on GPU.
- `PLMMSA_ALIGN_UVICORN_WORKERS` (default 4) exposed as an
  operator-visible env var in `.env` / `docker-compose.yml`.

### Changed

- `queue.embed_chunk_size` default raised 64 → 256. Combined with the
  length-descending sort in the orchestrator this keeps the GPU busy
  during target re-embed phases (especially OTalign + Ankh-Large,
  which has no shard store).
- `OrchestratorConfig.http_timeout` raised 120 s → 900 s. OTalign DP
  on a 235-residue query × ~1500 targets was hitting the 120 s cap.

### Fixed

- OTalign's Algorithm 1 step 5 filter was zeroing every hit because
  its transport-mass score scale (~[0, 1]) doesn't match the PLMAlign
  threshold. `aligners.otalign.filter_enabled` now defaults to false.
- Orchestrator drops alignment hits whose columns reference positions
  past the cached target sequence (symptom of `cache-seq` + shard
  store populated from different UniRef50 snapshots). Previously
  `render_hit` raised `IndexError` and failed the whole job. Drop
  count surfaces in the worker log at WARN.

### Added (this round)

- **Redis-backed shard path index.** `shard:<model>:<id> → folder`
  in `cache-seq`, populated once from the sqlite `index.db` by
  `python -m plmmsa.tools.build_shard_index`. Replaces the sqlite
  runtime lookup that paid ~13 s per chunk of 500 ids on `/gpfs`;
  MGET is now ~5 ms for 1500 ids. PLMAlign Phase 4 dropped ~35-50 s
  per job on the CASP15 benchmark.
- **Binary `/embed/bin` endpoint** on the embedding service with
  the orchestrator switched to it by default. JSON `/embed` kept
  for tests + legacy. Phase 4 parse time fell from ~200 s to ~10 s
  on OTalign k=1000 runs.
- **OTalign DP numba-JIT'd** (`_fill_matrices_jit` in
  `otalign_dp.py`). Phase 5 dropped 63 s → 24 s on T1104.
- **Paired MSA** via MMseqs-style taxonomy join. Per-chain
  retrieval at `paired_k = queue.paired_k_multiplier * k` (default
  3), `tax:UniRef50_<acc>` lookup, highest-scoring-per-chain-per-taxonomy
  selection, joint-score ranking. New `/v2/msa` accepts
  `paired=true`; new `assemble_paired_a3m` with gap separator
  `max(chain_len) // 10` (ColabFold convention). Stats block
  exposes `paired_rows`, `paired_taxonomies`, `paired_k_effective`.
- **ColabFold-compatible entrypoints** at
  `/v2/colabfold/{plmmsa,otalign}/*` — mirrors the MMseqs2 MsaServer
  wire shape so `colabfold_batch --host-url`, `boltz predict
  --msa_server_url`, and Protenix's MSA-server URL work as drop-ins.
  Each flavor hardwires its aligner; ticket id = plmMSA job id.
- **FlashSinkhorn** (`aligners.otalign.fused_sinkhorn`). `torch.compile`
  fused 16-iteration chunk with convergence check at chunk
  boundaries; dynamic shapes so no per-size recompilation.
  `_warmup_fused_sinkhorn` at align startup pays the compile cost
  once. Measured: 1.15-1.54× speedup on OTalign Phase 5 across
  CASP15 targets. `gcc/g++/make` added to the align image (Inductor
  builds Triton kernels at runtime).
- **Static submit-a-job web UI** at `/ui/`. Pure client — vanilla
  HTML/CSS/JS, no build step. BioJS msa viewer via CDN with
  graceful fallback. Query-param routing (`?job=<id>`,
  `?seq=<seq>`), localStorage job cache, adaptive 5 s / 60 s poll.
  Click the sequence label to copy the full sequence; hover for
  native tooltip.
- **`queue.paired_k_multiplier`** (default 3) — documented in
  `settings.example.toml`.

### Changed (this round)

- `ratelimit.per_ip_rpm` default raised 30 → 60. The 30-rpm cap was
  hitting active clients on legitimate retry + polling traffic.
