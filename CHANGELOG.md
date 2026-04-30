# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `POST /v2/templates/realign` — re-align an existing hmmsearch-style
  A3M against the query under OTalign / Ankh-Large / glocal. Output
  rows are exactly `query_len` chars from `[A-Z-]` (template residues
  with no matching query column are dropped, no lowercase A3M
  insertions). Headers preserve the original domain id + tail tokens
  and gain a re-intervalled `/start-end` plus a `score:N.NNN` token
  at the end of the technical-tokens section, before the description. Bearer-gated like `/v2/embed`. Per-request
  `sort_by_score` (default false) toggles between input-order and
  best-hit-first output. Operator walkthrough:
  [docs/templates-realign.md](./docs/templates-realign.md).
- `bin/realign_fixture.py` — one-shot operator script that runs the
  orchestrator end-to-end on the bundled CASP-style Exostosin fixture
  (593 records) and writes `tmp/exostosin_realigned.a3m` + a stats
  JSON. ~12 s on one Ada GPU.
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
  HTML/CSS/JS, no build step. Built-in colored MSA table viewer.
  Query-param routing (`?job=<id>`,
  `?seq=<seq>`), localStorage job cache, adaptive 5 s / 60 s poll.
  Click the sequence label to copy the full sequence; hover for
  native tooltip.
- **`queue.paired_k_multiplier`** (default 3) — documented in
  `settings.example.toml`.

### Changed (this round)

- `ratelimit.per_ip_rpm` default raised 30 → 60. The 30-rpm cap was
  hitting active clients on legitimate retry + polling traffic.

### Added (infra + observability round)

- **Per-service Prometheus `/metrics`.** The `MetricsMiddleware` + counters
  moved out of `plmmsa.api.metrics` into a shared `plmmsa.metrics`
  module. Every http-serving service (api / embedding / vdb / align)
  now mounts the same middleware + `/metrics` router with a
  `service="..."` label on every sample. Worker gets an embedded
  `prometheus_client.start_http_server` on `PLMMSA_WORKER_METRICS_PORT`
  (default 9090) exposing three worker-specific metrics:
  `plmmsa_worker_jobs_processed_total{status}`,
  `plmmsa_worker_pipeline_duration_seconds`,
  `plmmsa_worker_queue_depth` (sampled every 5 s from
  `LLEN plmmsa:queue`). Sidecar ports are bridge-only; scrape from a
  container on `plmmsa_net`.
- **Aggregated `/health` on api.** Fans out to `embedding`, `vdb`,
  `align` `/health` (2 s per-probe, 3 s overall cap) and PINGs every
  Redis role. Returns a per-service readiness map; overall status is
  `ok` only when every downstream is ok. Cached for ~1.5 s via an
  `asyncio.Lock`-guarded TTL so a burst of polls doesn't fan six
  probes per request. Bare `/healthz` is kept (and is now what the
  compose healthcheck hits) so downstream warmup can't cause compose
  to restart api.
- **Completed-MSA result cache** on `cache-emb`. Keyed by
  `sha256` of the canonicalized submit body (sequences normalized to
  uppercase + whitespace-stripped, chain order preserved,
  non-output-affecting fields like `force_recompute` and
  `request_id` dropped); value is the canonical A3M result. api
  checks the cache on submit and, on hit, synthesizes a `succeeded`
  Job record and returns its id without touching the queue
  (`result.stats.cache_hit = true`). Worker writes on job success
  best-effort (never fails the job on a cache outage). Clients can
  bypass with `{"force_recompute": true}`. Operator knobs:
  `PLMMSA_RESULT_CACHE_URL` (default `redis://cache-emb:6379`),
  `PLMMSA_RESULT_CACHE_TTL_S` (default 30 days).
- **Per-sidecar request-id middleware** on embedding / vdb / align.
  Adopt the api-supplied `X-Request-ID` (or mint one), bind it to a
  `ContextVar`, echo it on the response, and emit a structured
  access-log line under `plmmsa.access.{service}`. `api` now threads
  the id onto every downstream httpx call (already did on
  `/v2/embed|/search|/align`; now also from the orchestrator's
  internal calls). api also stamps `request_id` onto the Job
  payload; worker rebinds it on job claim so the orchestrator's
  httpx client carries it end-to-end. `request_id` is excluded from
  idempotency and result-cache key hashing so retries still dedup.
- **CSV ingestion for `build_sequence_cache`.** New `--csv` /
  `--csv-dir` modes that stream `uniref50_t5/split/*.csv` (columns
  `accession, description, sequence, length, length_group`) into
  `cache-seq`. TaxID is extracted from `description` via the same
  regex used on FASTA headers. `--start` / `--stop` slice the sorted
  file list for resume. `build_from_fasta` kept as a compat shim.
- **Redis container uid override.** New `REDIS_CONTAINER_USER` env
  (default `999:999`) on the three `cache-*` services in
  `docker-compose.yml`. On `deepfold` the host `.env` sets
  `220104:220104` so RDB/AOF files on
  `/gpfs/deepfold/service/cache_*_data` are owned by `deepfold:deepfold`.
- **Exception logging policy across services.** Every PlmMSAError
  handler (api / embedding / vdb / align) now logs with
  `logger.exception` on 5xx (full traceback) and `logger.warning` on
  4xx (code + message only), tagged with method + path. Orchestrator
  + shard-store fallbacks add `exc_info=True` so silent per-model /
  Redis failures carry tracebacks.
- **Host-path migration off `/store` onto `/gpfs`** (on the
  `deepfold` host). `.env` now points
  `MODEL_CACHE_DIR=/gpfs/deepfold/model_cache` and
  `CACHE_*_DATA_DIR=/gpfs/deepfold/service/cache_{ops,seq,emb}_data`
  (correct `service` spelling). `.env.example` gains the
  `REDIS_CONTAINER_USER`, `PLMMSA_RESULT_CACHE_URL`, and
  `PLMMSA_RESULT_CACHE_TTL_S` knobs.

### Removed (infra + observability round)

- **Dead per-residue embedding cache** branch in
  `plmmsa.embedding.server`. The `PLMMSA_EMBEDDING_CACHE_URL` path
  was disabled in a prior commit because 256-seq pipelines of
  pickled tensors overflowed the redis-py client buffer. The
  `cache-emb` container is now repurposed for the result cache
  (above); the dead code path, `hashlib`/`pickle` imports, and the
  `tests/test_embedding_cache.py` fixture are gone.
