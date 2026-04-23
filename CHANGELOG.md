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
