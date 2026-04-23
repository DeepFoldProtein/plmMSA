# plmMSA plan

Progress tracker for the rebase. Open issues / discussion live on the tracker;
this file is the check-list view of what shipped and what is next.

## Shipped

### M1 — bootable skeleton
- [x] Canonical repo foundation (LICENSE, NOTICE, CONTRIBUTING, CHANGELOG, .gitignore, `.github/` issue + PR templates).
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
See [`docs/maintenance.md`](./docs/maintenance.md#switching-faiss-index-size)
for the test-vs-full index swap recipe + a synthetic `/search` sanity check.
- [ ] Pick the index size in `settings.toml` `[vdb.collections.<name>].index_path`:
  - Quick validation: `{collection}_test.faiss` (~250 MB, loads in seconds).
  - Production: `{collection}_vdb.faiss` (~90 GB, needs ~100 GB RAM).
- [ ] `docker compose up -d vdb` and verify `GET /health` lists every
      enabled collection with the expected `dim` (Ankh=1536, ESM-1b=1280).
- [ ] Sanity-search with a synthetic 1536-dim (or 1280-dim) vector; expect
      `k` UniRef ids back.

### 4. Sequence + taxonomy cache (cache-seq)
See [`docs/maintenance.md#uniref50-sequence-cache`](./docs/maintenance.md#uniref50-sequence-cache).
`build_sequence_cache` emits two key spaces per record: `seq:{id}` → amino
acid sequence and `tax:{id}` → NCBI TaxID (parsed from the UniRef50 header).
- [ ] UniRef50 FASTA reachable (deepfold: `/gpfs/database/casp16/uniref50/uniref50.fasta`, 26 GB, ~60 M records).
- [ ] Pick an id scheme that matches the active FAISS index. UniRef50 FASTA
      keys are `UniRef50_*`; the legacy `_test.faiss` returns UniParc
      `UPI...`. Resolution options: rebuild FAISS over UniRef50 ids, or
      load a UniParc-keyed FASTA, or override `PLMMSA_SEQUENCE_KEY_FORMAT`.
- [ ] Run `build_sequence_cache` against the FASTA — `seq:*` + `tax:*`
      both populate into `cache-seq` (Redis URL from
      `PLMMSA_SEQUENCE_REDIS_URL`).
- [ ] Verify a handful of ids: `seq:{id}` returns the sequence, `tax:{id}`
      returns the TaxID, round-trippable via `RedisTargetFetcher`.

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
See [`docs/cloudflare-tunnel.md`](./docs/cloudflare-tunnel.md). The tunnel
runs as `cloudflared` behind the `tunnel` compose profile; it joins
`plmmsa_net` and forwards the public hostname → `api:8080`. Cloudflared
registers 4 QUIC connections to the nearest CF edges (on deepfold: ICN05 +
ICN06, Seoul/Incheon).
- [ ] Tunnel created in Zero Trust dashboard, `CLOUDFLARE_TUNNEL_TOKEN` in `.env`.
- [ ] Public hostname points to `api:8080` (confirmed by checking
      `docker compose logs cloudflared` for the registered ingress rule).
- [x] `docker compose --profile tunnel up -d` + end-to-end T1120 MSA via
      `https://plmmsa.deepfold.org/v2/msa` → 51-depth A3M,
      `hits_fetched = hits_found = 50`.
- [x] Every `/v2/*` and `/admin/*` route requires a bearer token. Only
      `/health`, `/v2/version`, `/openapi.json`, `/docs`, `/redoc`,
      and `/v1/*` (sunset) are anonymous.
- [ ] **CF dashboard: add a `/admin/*` ingress rule returning 404** before
      the `/` catch-all. Admin routes are bearer-gated but still
      path-reachable through the tunnel; adding the dashboard rule
      removes the surface entirely.
- [ ] Institutional sign-off on public availability.

### 8. Regression
- [x] `bench/run_regression.py --api-url http://localhost:8080 --k 500` —
      3/3 CASP15 fixtures pass against rebased baselines.
- [x] Depths within ±10% of the plmMSA `k=500` baseline in
      `expected_stats.json` (legacy depths kept as `legacy_msa_depth`
      for reference; not a like-for-like comparison).

## Discussion — design unknowns to settle before building

### Service capacity — how many jobs?

Open question: what's the per-host throughput target on `deepfold`
(2× RTX 6000 Ada, 48 GB each)?

Known cost profile from the April-2026 CASP15 benchmark
(`docs/submitting-msa.md` recipes, k=1000, both Ankh-CL + ESM-1b
retrieval):
- PLMAlign + ProtT5 shards (warm cache): ~30-60 s/job end-to-end,
  dominated by `/embed_by_id/bin` path resolution on sqlite (13+ s)
  → **being replaced by Redis path index** (expected ~5 ms).
- OTalign + Ankh-Large (live re-embed): ~20-40 s/job, dominated by
  Ankh-Large forward (~10 s Phase 4) + Sinkhorn+DP (~15-25 s Phase 5).

Rough steady-state capacity (single `worker` container,
`worker_concurrency = 4`):
- At 30 s/job × 4 concurrent = **~8 jobs/minute = ~480/hour**.
- `embedding` service is the shared bottleneck (Ankh-Large forward
  is serial per-request on one CUDA stream). Scaling horizontally
  means adding PLM replicas on more GPUs, not more worker containers.

Decisions to make:
- **Public rate limit vs. backpressure threshold.** Currently
  `queue.backpressure_threshold = 50` triggers `503 E_QUEUE_FULL`.
  At 30 s/job that is ~25 minutes of queued work — too long.
  Propose: drop to 20 (~10 min) or expose the wait-time estimate
  on the `POST /v2/msa` response.
- **Per-token priority lanes.** Authenticated clients (academic lab
  partners, example notebooks) may want guaranteed headroom.
  Simple: two Redis lists (`jobs:priority`, `jobs:standard`),
  worker drains priority first. Requires token-class metadata in
  the auth layer.
- **Scale-out story.** `docker compose up --scale worker=N` works
  for the worker, but multi-host embedding (Ankh-Large replica on
  a second machine) is an open topic. Until then, the single-host
  cap is the ceiling.

### plmMSA Web Server — static submit-a-job page

**Purely static HTML + JS.** Zero new server endpoints. State lives
in two places only: URL query parameters and `localStorage`. The
server stays a stateless JSON API; deployment is "copy the built
files to `services/api/public/`" and FastAPI's `StaticFiles` hands
them out.

Routing via query parameters (no hash router, no history API
juggling beyond `replaceState`):

- `/?seq=MKT...` — submit form pre-filled with that sequence.
- `/?job=<uuid>` — poll + render a single job (shareable link).
- `/?jobs=<uuid>,<uuid>,...` — dashboard view of an ad-hoc list.
- `/` — pull the cached job list from `localStorage` and render
  the same dashboard.

Behaviour:

- **Cache.** `localStorage.plmmsa.jobs` holds an array of
  `{id, submittedAt, label, lastStatus}` records. Every submission
  appends; every poll updates `lastStatus`. Never pruned
  automatically — a "clear" button exposes the list.
- **Refresh loop.** On load, iterate the cached list; for each
  non-terminal id, fire `GET /v2/msa/{id}`. Tab-visible → 5 s
  poll; tab-hidden → 60 s. Stop polling an id once status is
  terminal (succeeded / failed / cancelled). Exponential back-off
  on 5xx with a max of ~5 min.
- **Rendering.** On any terminal `succeeded`, pull `result.payload`
  (the A3M string), convert to FASTA client-side, feed to
  `msa-viewer` (https://github.com/intermine/msa-viewer). Download
  A3M button next to it. Stats (`hits_pre_filter`,
  `hits_post_filter`, `filter_applied`, `filter_threshold`) in a
  collapsed `<details>` block.
- **Sharing.** Copy-link writes `?job=<uuid>` to the URL bar. The
  recipient's static page reads the query and polls — no
  server-side sharing state needed.

Stack: vanilla TypeScript + Vite + pnpm. One page, no router, no
framework. Dev loop is `pnpm dev` against a running `api`; prod
output is a handful of static files under `services/api/public/`.

Explicitly out of scope: user accounts, server-side shared job
namespaces, personalization. If the page needs any of that, move it
behind the admin auth layer instead.

### Paired MSA generation + ColabFold drop-in API

Goal: ColabFold notebooks + Boltz / Protenix / AlphaFold pipelines
should be able to swap their MMseqs2 MSA server URL for plmMSA's and
get sensible results.

Upstream references to clone + study (into `.external/` during
development, gitignored — not vendored in the tree):

- **ColabFold MsaServer**: https://github.com/sokrypton/ColabFold
  (`MsaServer/` subdir). Flask app wrapping MMseqs2; defines the
  over-the-wire contract that's become the de-facto standard.
- **MMseqs2**: https://github.com/soedinglab/MMseqs2. The pairing
  logic lives in `src/workflow/PairAlign.cpp` (roughly): take
  per-chain MSAs, group hits by UniProt accession across chains,
  emit only rows where every chain has a hit under the same
  accession. Our equivalent needs the `seq:` Redis keyspace plus
  a `tax:` lookup, which `build_sequence_cache.py` already
  populates from UniRef50 headers.

Pairing method to document (write-up → `docs/paired-msa.md`),
modelled on MMseqs2's `PairAlign`:

1. Run per-chain plmMSA with a **larger `k`** than unpaired requests
   use. Taxonomy-pairing is a filter, so the per-chain pool needs
   headroom. Target a `paired_k = multiplier * effective_k` (start
   with `multiplier = 3`; tune). Capped at `limits.max_k` so
   operators can keep the GPU load bounded.
2. For each chain's hits, resolve `hit_id → (uniprot_accession,
   taxonomy_id)`. UniRef50 cluster representatives carry this in
   their FASTA header (`TaxID=<n>`); we already persist it as
   `tax:UniRef50_<acc>` via `build_sequence_cache.py`.
3. **MMseqs `PairAlign` algorithm** (translate from
   `MMseqs2/src/workflow/PairAlign.cpp`): bucket hits by taxonomy id;
   for each taxonomy that appears in every chain, pick the
   **highest-scoring representative per chain**; emit one paired
   row per shared taxonomy. Drop taxonomies that miss any chain.
4. Output one paired A3M: each row concatenates the chain-aligned
   sequences, separated by a deliberate gap run (length =
   `max(chain_lengths) // 10`, matches ColabFold convention).
5. **Rank by joint score** — sum of per-chain alignment scores for
   each paired row. Optional joint threshold — reuse the per-aligner
   `filter_enabled` first, add a paired-specific knob only if
   calibration demands it.

Scope: **pair on UniRef50 representatives only.** The taxonomy
stored in `tax:UniRef50_<acc>` (from the cluster representative's
FASTA header) is authoritative — no cluster-member expansion, no
UniParc lookups, no supplementary taxonomy sources. Simpler, and it
keeps the pairing surface identical to retrieval's id namespace.

Open question:
- **Multiplier calibration.** Start at `paired_k = 3 × k`; rerun the
  CASP15 multimer fixtures with 2×, 5×, 10× and pick the knee.
  Target deepfold2's paired coverage as the calibration anchor.

### Downstream-model integration guide

All three engines accept an **MSA API URL argument** — the
integration is "point the engine at our host" rather than "write a
custom A3M fetcher per engine". Sibling repo (`plmmsa-examples`)
remains the primary target per CLAUDE.md; the in-repo docs cover the
minimum drop-in recipe — just the flag / config key to set.

Write-up → `docs/integrations/`:

- **ColabFold** (https://github.com/sokrypton/ColabFold). The
  `colabfold_batch` CLI takes `--host-url <URL>`. Recipe:
  `colabfold_batch --host-url https://plmmsa.deepfold.org/v2/colabfold/plmmsa ...`
  (or `.../otalign`). `colabfold_batch` appends the MsaServer route
  (`/ticket/msa`, etc.) to whatever base URL you pass, so our
  compat routers sit under the `/v2/` namespace with one sub-path
  per aligner flavor — see below.
- **Boltz** (https://github.com/jwohlwend/boltz). `boltz predict`
  CLI takes `--use_msa_server --msa_server_url <URL>`. Recipe
  (PLMAlign flavor): `boltz predict --use_msa_server --msa_server_url https://plmmsa.deepfold.org/v2/colabfold/plmmsa ...`.
  Swap `.../plmmsa` for `.../otalign` to use OTalign instead.
  (Boltz speaks the ColabFold MsaServer wire shape, so the
  compat entrypoint is the right target.)
- **Protenix** (https://github.com/bytedance/Protenix). Exposes an
  MSA server URL in its inference config — also targets the
  ColabFold MsaServer wire shape, so the plmMSA URL to set is
  `https://plmmsa.deepfold.org/v2/colabfold/plmmsa` (or
  `.../otalign`). Confirm exact config key when writing up the
  recipe (clone into `.external/`).

For each: one short README, one example invocation, screenshot.
Target "paste-the-flag" level; deeper integration questions go to
the upstream project's issues.

**Dev workflow**: clone each repo into `.external/` (gitignored)
during write-up to confirm the exact flag name / config key.
Do **not** vendor them — we link to upstream and document only.

### ColabFold-compatible entrypoints — `/v2/colabfold/{plmmsa,otalign}/*`

`colabfold_batch --host-url` and the ColabFold notebook talk the
MMseqs2 MsaServer wire shape. Our native `/v2/msa` carries plmMSA-
specific fields (aligner, mode, score_model, filter), so we ship
**two parallel namespaces** that speak CF's exact shape, each
hardwiring a different aligner internally:

- `/v2/colabfold/plmmsa/*` — PLMAlign (our default; fast via ProtT5
  shard store).
- `/v2/colabfold/otalign/*` — OTalign (Sinkhorn + position-specific
  DP, Ankh-Large scoring; slower but upstream's default pairing).

Clients pick the flavor by setting `--host-url`; CF doesn't know or
care which is which. This also means we don't have to stuff aligner
selection into CF's form schema (it can't carry it anyway).

**Routes per namespace** (`<flavor>` ∈ `plmmsa` | `otalign`,
match ColabFold's MsaServer verbatim — reference at
https://github.com/sokrypton/ColabFold/tree/main/MsaServer):

- `POST /v2/colabfold/<flavor>/ticket/msa` — CF form-encoded body
  (`q`, `mode`, `database`). Returns
  `{"id": <ticket>, "status": "PENDING"}`.
- `POST /v2/colabfold/<flavor>/ticket/pair` — same shape, paired
  multimer.
- `GET /v2/colabfold/<flavor>/ticket/msa/<ticket>` — status.
  Returns `{"id", "status"}` with
  `PENDING | RUNNING | COMPLETE | ERROR | UNKNOWN`.
- `GET /v2/colabfold/<flavor>/result/download/<ticket>` — tar
  matching CF's expectation: `uniref.a3m` (unpaired) +
  `pair.a3m` (paired, otherwise empty) +
  `bfd.mgnify30.metaeuk30.smag30.a3m` (we emit the plmMSA A3M
  under that name for compat).

**Implementation** — one router, two registrations:

- `plmmsa.api.routes.colabfold` — factory function
  `make_router(aligner: str) -> APIRouter`. Each call returns a
  router that hardwires the chosen aligner in its `SubmitRequest`
  builder; the routing handlers themselves are identical.
- App wiring:
  ```python
  app.include_router(make_router("plmalign"),
                     prefix="/v2/colabfold/plmmsa", tags=["colabfold"])
  app.include_router(make_router("otalign"),
                     prefix="/v2/colabfold/otalign", tags=["colabfold"])
  ```
- **Translation layer**, not a separate pipeline. Each CF endpoint
  reuses the existing `/v2/msa` job lifecycle:
  - `POST .../ticket/msa` → parse CF body → build our
    `SubmitRequest` with `aligner=<flavor-aligner>`, server-default
    models + score_model + filter_by_score → `JobStore.enqueue`
    → respond `{id: <job_id>, status: "PENDING"}`.
  - `GET .../ticket/msa/<id>` → look up job → map status
    `{queued → PENDING, running → RUNNING, succeeded → COMPLETE,
    failed/cancelled → ERROR}`.
  - `GET .../result/download/<id>` → fetch `JobResult.payload`
    (A3M) → wrap in CF's tar layout → stream as
    `application/octet-stream`.
- **Ticket id = plmMSA job id.** Reuses existing storage; no
  second job table. The flavor is encoded in the job record via
  the stamped aligner — status / download handlers don't need to
  know which flavor claimed the id.
- **`database` arg accepted but ignored.** CF's `uniref` / `envdb`
  maps to our aggregate VDBs regardless. Log at INFO when a
  non-default value appears.

**Client recipes**:

```sh
# PLMAlign (default, fast)
colabfold_batch --host-url https://plmmsa.deepfold.org/v2/colabfold/plmmsa \
    input.fasta outdir/

# OTalign (Sinkhorn + position-specific DP)
colabfold_batch --host-url https://plmmsa.deepfold.org/v2/colabfold/otalign \
    input.fasta outdir/
```

Same `--host-url` swap works for `boltz predict --msa_server_url` and
Protenix's MSA-server config key.

**Auth**. CF's public MsaServer is anonymous; both
`/v2/colabfold/*` namespaces mirror that — same rate-limit story
as `/v2/msa` (Cloudflare + slowapi).

**Paired MSAs**. `POST .../ticket/pair` only becomes useful once
our paired-MSA path ships. Scaffold now with a `501 Not Implemented`
body explaining the gap; wire the real handler when paired pairing
lands.

**Testing**. The contract is the CF wire shape, so the fixture is a
captured `colabfold_batch` submission. Record one interaction per
flavor (request + expected response) and replay through FastAPI's
`TestClient` — that's the regression we gate on.

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
- [ ] GHCR image releases on tag (manual `docker push` until a CI
      workflow is added back).
- [ ] **Finish pLM-BLAST acceleration.** Numba-JIT landed for the DP
      fill (matches PLMAlign's 1 ms/target on small matrices), but a
      32-target benchmark with the thread-pool fanout took >5 min —
      likely numba dispatcher contention on first-compile or
      per-task pool overhead swamping the gain. Profile + pick a
      remedy: (a) eager-compile via explicit signature strings so no
      thread races the JIT, (b) switch the per-target fanout to
      `numba.prange` inside a jitted outer loop, (c) port the whole
      pipeline to torch. Default aligner stays `plmalign` (fast,
      JIT'd) until plm_blast is demonstrably competitive.
- [ ] **Cross-PLM scoring** (`aligners.plmalign.score_model`). Today the
      score matrix is built from the *same* PLM that searched the VDB; this
      entry lands an operator-configurable override that re-embeds the
      query + targets with a different PLM before scoring. Requires an
      orchestrator rework: after the VDB-search step, the worker looks up
      `score_model` in the enabled PLM set and routes an extra embed pass
      (with shard-store shortcut when available). Cost: one extra embed
      round-trip per hit in the non-cached path; zero when the
      `score_model` has a shard store mount. Useful for "search cheap,
      score with a bigger model" deployments.
- [ ] **Precomputed-shard stores for Ankh family.** ProtT5 shards already
      land via `settings.models.prott5.shard_root`; a matching store for
      Ankh-Large / Ankh-CL would cut target-embedding cost on the default
      aggregate path. Requires a one-time offline pass over UniRef50 to
      produce per-sequence `.pt` files + an SQLite index; reuses the
      existing `ShardStore` reader unchanged.

## Open-source readiness

- [x] `procl` reachable (vendored in `src/procl/`).
- [x] Repo license MIT; upstream PLMAlign attribution preserved in `NOTICE`.
- [x] Secret hygiene: `.env` / `settings.toml` gitignored; `.example` files use `localhost` defaults.
- [ ] Ankh-CL weight distribution decision: public HF repo (current default) vs. token-gated fetch.
- [ ] ProtT5 (RostLab custom license) attribution on the web frontend.
- [ ] Institutional sign-off from Prof. Joo before flipping the tunnel hostname to production traffic.
- [x] Abuse surface: per-IP rate limit configurable in `settings.toml`, `/admin/*` not tunnel-exposed.
