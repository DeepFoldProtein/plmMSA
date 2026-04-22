# plmMSA plan

Progress tracker for the rebase. Open issues / discussion live on the tracker;
this file is the check-list view of what shipped and what is next.

## Shipped

### M1 — bootable skeleton
- [x] Canonical repo foundation (LICENSE, NOTICE, CONTRIBUTING, CHANGELOG, .gitignore, `.github/` CI + templates).
- [x] `uv` + `pyproject.toml`, Python 3.11, `src/plmmsa/` layout.
- [x] Configuration: `settings.toml` (non-secret tunables) + `.env` (secrets + host paths), `*.example` committed.
- [x] Error taxonomy (`plmmsa.errors`) with stable codes: `E_SEQ_TOO_LONG`, `E_GPU_OOM`, `E_QUEUE_FULL`, `E_AUTH_*`, `E_CANCELLED`, …
- [x] FastAPI `api` service with `/health`, `/v2/version`, `/openapi.json`, CORS middleware, PlmMSA error handler.
- [x] URL versioning: `/v1/*` returns `410 E_GONE` with sunset date; `/v2/*` current.
- [x] docker-compose skeleton: 6 services (`api`, `embedding`, `vdb`, `align`, `worker`, `cache`) on `plmmsa_net` bridge.
- [x] `bin/{up,down,logs}.sh` bootstrap scripts.
- [x] Host paths configurable via `.env`: `MODEL_CACHE_DIR`, `VDB_DATA_DIR`, `CACHE_DATA_DIR`.

### M2 — embedding service
- [x] `plmmsa.plm.base.PLM` ABC (per-residue encode contract).
- [x] Concrete backends: `AnkhCL`, `AnkhLarge`, `ESM1b`, `ProtT5`.
- [x] `plmmsa.plm.registry` with failure-tolerant loader.
- [x] `src/procl/` vendored (cleaned copy of DeepFoldProtein/DeepFold-PLM/plmMSA/src/procl).
- [x] `plmmsa.embedding` FastAPI app (`create_app` factory, `/health`, `POST /embed`, GPU-OOM detection).
- [x] Real `services/embedding/Dockerfile` (python:3.11-slim + torch 2.8 / CUDA 12.1 wheel).
- [x] `api:/v2/embed` bearer-gated passthrough to the embedding service.
- [x] Slow integration test — real Ankh-CL load + encode (`RUN_SLOW=1 uv run pytest tests/test_plm_ankh_cl.py`).

### M3 — VDB service
- [x] `plmmsa.vdb.base.VDB` ABC (batched k-NN contract).
- [x] `plmmsa.vdb.faiss_vdb.FaissVDB` — loads `*.faiss` + `*.pkl` id mapping from `VDB_DATA_DIR`.
- [x] `plmmsa.vdb.registry` (per-collection failure tolerant).
- [x] `plmmsa.vdb` FastAPI app (`/health`, `POST /search`).
- [x] `services/vdb/Dockerfile` with faiss-cpu.
- [x] `api:/v2/search` bearer-gated passthrough.
- [x] Settings: `[vdb.collections.*]` including `ankh_uniref50` + `esm1b_uniref50`.

### M4 — align service
- [x] `plmmsa.align.base.Aligner` ABC with `**kwargs` passthrough.
- [x] Clean PLMAlign (fresh impl, not a port): affine-gap Smith-Waterman / Needleman-Wunsch over cosine similarity, numpy-only.
- [x] `plmmsa.align.otalign` scaffold — raises `E_NOT_IMPLEMENTED 501`.
- [x] `plmmsa.align.registry` + `plmmsa.align` FastAPI app (`/health`, `POST /align`).
- [x] `services/align/Dockerfile` (CPU-only).
- [x] `api:/v2/align` bearer-gated passthrough.

### M5 — async job lifecycle + real pipeline
- [x] `plmmsa.jobs.{models,store}` — `Job`, `JobStatus`, `JobResult`, `JobError`, `JobStore` (redis.asyncio).
- [x] `api:/v2/msa` POST enqueues, GET returns record, DELETE cancels.
- [x] `plmmsa.worker` process with graceful SIGTERM drain.
- [x] `plmmsa.pipeline`: `Orchestrator` (embed → search → fetch → embed → align → assemble), `TargetFetcher` ABC, `AlignmentHit` + `assemble_a3m` + `render_hit`.
- [x] Fetcher impls: `DictTargetFetcher`, `FastaTargetFetcher`, `RedisTargetFetcher`.
- [x] `plmmsa.tools.build_sequence_cache` CLI — streaming FASTA → Redis.
- [x] `services/worker/Dockerfile`.
- [x] Fixture round-trip test against upstream A3M example (byte-for-byte).

### M6 — operations
- [x] Token admin: `plmmsa.admin.tokens.TokenStore` (Redis-backed, SHA-256 hashed), `/admin/tokens` POST/GET/DELETE, bearer gate accepts bootstrap `ADMIN_TOKEN` OR minted store tokens.
- [x] Cloudflare Tunnel: `cloudflared` service under `profiles: [tunnel]`, `CLOUDFLARE_TUNNEL_TOKEN` env. Walkthrough in [`docs/cloudflare-tunnel.md`](./docs/cloudflare-tunnel.md).
- [x] CASP15 regression harness: `tests/fixtures/casp15/{T1104,T1120,T1132}.fasta` + `expected_stats.json` + `bench/run_regression.py`.
- [x] Maintenance runbook: [`docs/maintenance.md`](./docs/maintenance.md) — start/stop, prereqs, host-path knobs, cache clearing, token rotation, failure modes.
- [x] Stack smoke: `./bin/up.sh` with CPU-only services validated end-to-end (queue + worker + error propagation + admin-token mint/use).

## Service readiness — before the first real MSA goes out

Runbook for flipping the stack from "scaffolded" to "producing MSAs on UniRef50".

### 1. Host prerequisites
- [ ] `nvidia-container-toolkit` installed (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must succeed).
- [ ] Writable host dirs exist with correct ownership:
  - [ ] `MODEL_CACHE_DIR` (multi-GB; Ankh-CL alone is ~3 GB)
  - [ ] `VDB_DATA_DIR`
  - [ ] `CACHE_DATA_DIR` (Redis persistence)
- [ ] `.env` filled in: GPU device pins, `ADMIN_TOKEN` rotated away from default.

### 2. Model weight warmup
See [`docs/warming-weights.md`](./docs/warming-weights.md) for the
concrete `hf download` commands.
- [ ] Point `MODEL_CACHE_DIR` in `.env` at the host's shared HF cache tree
      (e.g. `/store/deepfold/huggingface` on the deepfold host).
- [ ] `export HF_HOME=$MODEL_CACHE_DIR` and `uv run hf download DeepFoldProtein/Ankh-Large-Contrastive` (plus any other PLMs to enable).
- [ ] In `settings.toml`, set `[models.*].enabled` for only the PLMs you want resident at startup.
- [ ] `docker compose up -d embedding` and confirm `GET /health` lists the resident PLMs with their device pins.

### 3. FAISS index
- [ ] Pick the index size:
  - For quick validation: `{collection}_test.faiss` (~250 MB) — point `[vdb.collections.ankh_uniref50].index_path` at it.
  - For production: the full `{collection}_vdb.faiss` (~91 GB) — needs ~100 GB RAM to load.
- [ ] Confirm VDB_DATA_DIR mount + index paths resolve from inside the `vdb` container (`docker compose logs vdb` on startup; FaissVDB rejects dim-mismatched indexes loudly).
- [ ] Each enabled collection must match a PLM's embedding dim (`dim` field in settings).

### 4. Sequence cache
- [ ] UniRef50 FASTA available at a path operators can reach (`/gpfs` or a local download).
- [ ] Populate once, then leave:
  ```bash
  uv run python -m plmmsa.tools.build_sequence_cache \
      --fasta /path/to/uniref50.fasta \
      --redis-url redis://localhost:6379
  ```
- [ ] Verify a few ids round-trip via `RedisTargetFetcher` (see the example in `docs/maintenance.md`).

### 5. Tokens + auth
- [ ] `ADMIN_TOKEN` in `.env` rotated off `change-me`.
- [ ] Mint per-client tokens via `POST /admin/tokens` (curl recipe in `docs/maintenance.md`).
- [ ] Document which client owns which token (`label` field).
- [ ] Decide whether to keep the bootstrap token as an active credential or retire it once minting is done.

### 6. Bring it all up
- [ ] `./bin/up.sh` — all 6 services healthy (incl. `embedding` and `vdb`, which smoke skipped).
- [ ] `curl http://localhost:8080/health` → ok.
- [ ] `curl http://localhost:8080/v2/version` lists enabled models.
- [ ] Submit T1120 from the regression fixtures; job succeeds; `result.format=a3m`, `result.stats.hits_fetched > 0`.

### 7. Public edge (Cloudflare Tunnel)
- [ ] Tunnel created in Zero Trust dashboard, `CLOUDFLARE_TUNNEL_TOKEN` in `.env`.
- [ ] Public hostname points to `api:8080` ONLY — never `/admin/*`, never any other service.
- [ ] `docker compose --profile tunnel up -d` and verify `curl https://plmmsa.deepfold.org/health`.
- [ ] Institutional sign-off on public availability.

### 8. Regression
- [ ] `bench/run_regression.py --api-url http://localhost:8080` against all CASP15 fixtures.
- [ ] Depths within tolerance of the legacy baseline in `expected_stats.json`. If not, investigate before opening up the tunnel.

## Deferred — pick up after the first real MSA ships

- [ ] Paired MSA across chains with UniProt-ID → taxonomy lookup (sibling Redis or shared cache).
- [ ] Per-token rate limits + Prometheus-format `/metrics` per service.
- [ ] Request-ID propagation + structured JSON logs across services.
- [ ] Result cache (`plmmsa:result:*`) with size + age eviction.
- [ ] Output formats beyond A3M (Stockholm, paired variants).
- [ ] OTalign real implementation (currently raises 501).
- [ ] Backpressure: `503 Retry-After` when queue depth exceeds threshold.
- [ ] Internal admin UI (HTML; the JSON API under `/admin/*` is the underlying transport).
- [ ] External web frontend (submit-a-job page, thin client over `/v2/*`).
- [ ] Sibling `plmmsa-examples` repo: ColabFold notebook + AlphaFold / OpenFold / Boltz / Protenix integration recipes.
- [ ] GHCR image releases on tag (CI workflow extension).

## Open-source readiness

- [x] `procl` reachable (vendored in `src/procl/`).
- [x] Repo license MIT; upstream PLMAlign attribution preserved in `NOTICE`.
- [x] Secret hygiene: `.env` / `settings.toml` gitignored; `.example` files use `localhost` defaults.
- [ ] Ankh-CL weight distribution decision: public HF repo (current default) vs. token-gated fetch.
- [ ] ProtT5 (RostLab custom license) attribution on the web frontend.
- [ ] Institutional sign-off from Prof. Joo before flipping the tunnel hostname to production traffic.
- [x] Abuse surface: per-IP rate limit configurable in `settings.toml`, `/admin/*` not tunnel-exposed.
