# plmMSA maintenance

Runbook for keeping the stack healthy on the `deepfold` host. Pair with
`docs/cloudflare-tunnel.md` for the public-edge piece and
`docs/submitting-msa.md` for the client-facing API recipe.

## Validating a running stack

One-shot smoke against the public hostname (or substitute `http://localhost:8080`
for a local-only run). Covers every surface a client hits:

```bash
BASE=https://plmmsa.deepfold.org
BOOT=$(grep '^ADMIN_TOKEN=' .env | cut -d= -f2-)

# 1. Health + version (unauthenticated).
curl -sS $BASE/health
curl -sS $BASE/v2/version | jq .models

# 2. /v1/* is intentionally 410 Gone.
curl -sS $BASE/v1/anything

# 3. Auth gate — /v2/embed|/search|/align and /admin/* require a bearer token.
curl -sS -X POST $BASE/v2/embed -d '{"model":"ankh_cl","sequences":["MKT"]}' \
    -H 'Content-Type: application/json'   # expect 401 E_AUTH_MISSING

# 4. (Optional) Mint a client token for the raw-service passthroughs.
TOKEN=$(curl -sS -X POST $BASE/admin/tokens \
    -H "Authorization: Bearer $BOOT" -H 'Content-Type: application/json' \
    -d '{"label":"smoke"}' | jq -r .token)

# 5. End-to-end MSA submission + poll. /v2/msa is anonymous; omitting
#    `models` aggregates every PLM with a VDB collection (ankh_cl + esm1b).
SEQ=$(tail -n +2 tests/fixtures/casp15/T1120.fasta | tr -d '\n')
JID=$(curl -sS -X POST $BASE/v2/msa \
    -H 'Content-Type: application/json' \
    -d "$(jq -Rn --arg s "$SEQ" '{sequences:[$s],query_id:"T1120",k:50}')" \
    | jq -r .job_id)
while true; do
    ST=$(curl -sS $BASE/v2/msa/$JID | jq -r .status)
    echo "status: $ST"
    [[ "$ST" == "succeeded" || "$ST" == "failed" || "$ST" == "cancelled" ]] && break
    sleep 5
done

# 6. Expected: status=succeeded, stats.hits_fetched == stats.hits_found, depth = k + 1.
```

Protected routes (every one returns 401 E_AUTH_MISSING without a bearer token):

- `POST /v2/embed`, `POST /v2/search`, `POST /v2/align`
- `POST /admin/tokens`, `GET /admin/tokens`, `DELETE /admin/tokens/{id}`

Unauthenticated:

- `POST /v2/msa`, `GET /v2/msa/{id}`, `DELETE /v2/msa/{id}` — open on
  purpose so clients can submit without minting a token. Abuse is bounded
  by the per-IP rate limiter + queue backpressure. Drive-by cancellation
  is mitigated by UUID4 job ids (not enumerable).
- `POST/GET /v2/colabfold/{plmmsa,otalign}/*` — ColabFold-compat
  entrypoints (mirrors MMseqs2 MsaServer shape). Same anonymous
  posture as `/v2/msa`; drop-in target for `colabfold_batch --host-url`,
  `boltz predict --msa_server_url`, and Protenix's MSA-server config.
- `GET /ui/*` — static submit-a-job web UI. Pure client; no new
  server-side state. Users bookmark `<host>/ui/`.
- `/health`, `/v2/version`, `/metrics`, `/v1/*` (sunset), `/openapi.json`,
  `/docs`, `/redoc`.

## Starting / stopping the stack

```bash
cp .env.example .env                         # first-time only
cp settings.example.toml settings.toml       # first-time only
./bin/up.sh                                  # full stack (all 6 services)
./bin/down.sh                                # graceful stop; keeps state volumes
./bin/down.sh -v                             # stop AND delete persistent volumes (destructive)
./bin/logs.sh <service>                      # tail logs, e.g. `./bin/logs.sh worker`
```

`./bin/up.sh` fails fast if `.env` or `settings.toml` are missing. All six
services (`api`, `embedding`, `vdb`, `align`, `worker`, `cache`) come up on
the `plmmsa_net` bridge. To add the public edge, use
`docker compose --profile tunnel up -d`.

## Prerequisites on a fresh host

- Docker ≥ 24 with Compose v2.
- `nvidia-container-toolkit` installed and `nvidia-ctk runtime configure`'d —
  the `embedding` service requests `nvidia` GPU reservations through compose.
  Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
- A writable `CACHE_DATA_DIR` (Redis persistence) and `MODEL_CACHE_DIR`
  (PLM weight cache). `./bin/up.sh` does not create these — if the path in
  `.env` doesn't exist, compose will try to create it under `root`, which is
  usually wrong. Create them yourself with appropriate ownership first.

## Controlling where state lives

`.env` keys that determine on-host paths:

| Key                  | Default            | Used by                                       |
| -------------------- | ------------------ | --------------------------------------------- |
| `MODEL_CACHE_DIR`    | `./model_cache`    | `embedding` — PLM weights (HuggingFace cache) |
| `VDB_DATA_DIR`       | `./vdb_data`       | `vdb` — FAISS index + id-mapping pickles      |
| `CACHE_OPS_DATA_DIR` | `./cache_ops_data` | `cache-ops` — jobs, tokens, rate limits (AOF) |
| `CACHE_SEQ_DATA_DIR` | `./cache_seq_data` | `cache-seq` — `seq:*` + `tax:*` lookups (RDB) |
| `CACHE_EMB_DATA_DIR` | `./cache_emb_data` | `cache-emb` — PLM embedding cache (RDB + LRU) |

Point these at `/gpfs` or `/store` for multi-machine persistence; keep them
local for a single-host dev deployment.

## Three Redis instances, one per role

| Instance    | Image            | Holds                                          | Persistence    | Eviction                                           | Typical size                    |
| ----------- | ---------------- | ---------------------------------------------- | -------------- | -------------------------------------------------- | ------------------------------- |
| `cache-ops` | redis:7.4-alpine | `plmmsa:job:*`, `plmmsa:queue`, `admintoken:*` | AOF on, no RDB | `noeviction`                                       | MBs                             |
| `cache-seq` | redis:7.4-alpine | `seq:*`, `tax:*` (UniRef50 + taxonomy)         | RDB snapshots  | `noeviction` (configurable via `CACHE_SEQ_POLICY`) | ~30 GB                          |
| `cache-emb` | redis:7.4-alpine | PLM embedding cache entries                    | RDB snapshots  | `allkeys-lru`                                      | capped by `CACHE_EMB_MAXMEMORY` |

Why three instead of one:

- **Eviction.** `cache-emb` can evict safely (embeddings are rebuildable);
  `cache-ops` must never evict (you'd lose a queued job or revoke a token
  by accident). A single Redis can't have two policies at once.
- **Restart time.** `cache-ops` reloads in a second and the API is live;
  `cache-seq` takes minutes to replay the UniRef50 RDB but nothing
  user-facing depends on it at that moment.
- **Blast radius.** A bug that FLUSHDBs the wrong instance is less bad
  when roles are isolated.

## Clearing / wiping

Per-role, non-destructive:

```bash
# Drop just the sequence cache (keep queue + tokens + embedding cache).
docker compose exec cache-seq redis-cli --scan --pattern 'seq:*' | \
    xargs -n 500 docker compose exec cache-seq redis-cli del

# Drop just the embedding cache.
docker compose exec cache-emb redis-cli FLUSHDB
```

Per-instance wipe (fast restart when AOF/RDB growth hurts):

```bash
./bin/down.sh
rm -rf "$(grep '^CACHE_SEQ_DATA_DIR=' .env | cut -d= -f2-)"/*
./bin/up.sh
# Re-run build_sequence_cache to repopulate.
```

**Why "clear `cache_seq_data` for fast loading" matters:** a 30 GB RDB
takes 1–3 minutes to replay on Redis startup. If you're iterating on
schema / id-format changes, wiping the data dir between runs is cheaper
than a BGREWRITE. The sequence + tax data is trivially rebuildable from
the source FASTA:

```bash
uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /gpfs/database/casp16/uniref50/uniref50.fasta \
    --redis-url redis://localhost:6379
```

## Rotating the bootstrap token

`.env` holds `ADMIN_TOKEN` — the bootstrap credential used to mint per-client
tokens via `/admin/tokens`. To rotate:

1. Generate a new value: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
2. Replace `ADMIN_TOKEN=` in `.env`.
3. `docker compose up -d api worker` to pick up the new env.
4. Re-mint any dependent per-client tokens; old minted tokens stay valid
   (they live in Redis, not the env), but the old bootstrap stops working.

## Managing per-client tokens

Only the bootstrap token or another live admin token can call these:

```bash
T="$(grep '^ADMIN_TOKEN=' .env | cut -d= -f2-)"

# Mint a new client token. Plaintext is shown ONCE.
curl -sX POST http://localhost:8080/admin/tokens \
    -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
    -d '{"label":"colab-notebook"}'

# List tokens.
curl -s http://localhost:8080/admin/tokens -H "Authorization: Bearer $T"

# Revoke a token by its id (not the plaintext).
curl -sX DELETE http://localhost:8080/admin/tokens/<token_id> \
    -H "Authorization: Bearer $T"
```

Revocation is immediate: the hash is still reachable for audit but
`verify()` returns `None` once `revoked=true`.

**Never expose `/admin/*` through the public Cloudflare tunnel.** The
tunnel hostname must only target `/health`, `/v1/`, `/v2/`, `/openapi.json`,
`/docs`, and `/redoc`. See `docs/cloudflare-tunnel.md`.

## Rebuilding images

After a dependency change (`pyproject.toml`) or a code change that has to
ship in an image (rare — services read `src/` off a bind-like COPY at build
time, so `./bin/up.sh` rebuilds when the context changes), just:

```bash
docker compose build               # all services
docker compose build embedding     # one service
./bin/up.sh                        # up does a rebuild too
```

The `embedding` image is large (~7.2 GB) because it bundles torch + CUDA
runtime. `uv`'s layer-cached install keeps rebuilds fast as long as
`pyproject.toml` / `uv.lock` don't move.

## Populating model weights

PLM checkpoints are warmed **on the host**, not inside the `embedding`
container. The container mounts `MODEL_CACHE_DIR` read-only with
`HF_HUB_OFFLINE=1` — it cannot download. See the step-by-step in
[`docs/warming-weights.md`](./warming-weights.md).

Short version:

```bash
export HF_HOME="$(grep '^MODEL_CACHE_DIR=' .env | cut -d= -f2-)"
uv run hf download DeepFoldProtein/Ankh-Large-Contrastive   # ankh_cl backend
uv run hf download ElnaggarLab/ankh-large                   # shared tokenizer
docker compose up -d embedding
```

## UniRef50 sequence cache

Step 4 of `PLAN.md`. The orchestrator's fetch stage pulls target sequences
from `cache-seq` (Redis) keyed as `seq:{id}`, and the companion taxonomy
lookup from `tax:{id}`. Populate once per host; the keys live on
`$CACHE_SEQ_DATA_DIR` and survive `./bin/down.sh` / `up.sh`.

### On the deepfold host

The UniRef50 FASTA lives at:

```
/gpfs/database/casp16/uniref50/uniref50.fasta   # 26 GB, ~60 M sequences
```

### Quick subset validation (seconds)

```bash
head -50000 /gpfs/database/casp16/uniref50/uniref50.fasta > /tmp/uniref50_head.fasta
docker compose up -d cache-seq worker
docker compose cp /tmp/uniref50_head.fasta worker:/tmp/uniref50_head.fasta
docker compose exec -T worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /tmp/uniref50_head.fasta --redis-url redis://cache-seq:6379 --batch 200
# spot-check a few ids for both seq and tax
for id in $(docker compose exec -T cache-seq redis-cli --scan --pattern 'seq:UniRef50_*' | head -3); do
    name="${id#seq:}"
    echo -n "$name  seq len: "; docker compose exec -T cache-seq redis-cli STRLEN "seq:$name"
    echo -n "$name  tax id:  "; docker compose exec -T cache-seq redis-cli GET "tax:$name"
done
```

### Full load

```bash
docker compose up -d cache-seq worker
docker compose run --rm \
    -v /gpfs/database/casp16/uniref50:/uniref-src:ro \
    worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /uniref-src/uniref50.fasta \
    --redis-url redis://cache-seq:6379 \
    --batch 5000
```

On the deepfold host this runs at ~50 k keys/sec (measured on the
2-GPU deepfold host, writing to the `/store` NFS mount) — ~20 minutes for
the full ~60 M UniRef50 records. Every record emits both a `seq:*` and a
`tax:*` key, so final DBSIZE is ~120 M.

### ID-format caveat

`seq:{id}` is only useful if `{id}` matches what the VDB returns on
`/search`. The legacy test FAISS at `ankh_uniref50_test.faiss` returns
**UniParc `UPI...` ids**, whereas UniRef50 FASTA records use
`UniRef50_...` ids — those two id spaces don't overlap. Three options,
pick one:

1. **Rebuild the FAISS index over UniRef50-ID-keyed embeddings.** The
   clean long-term answer.
2. **Load a UniParc-keyed FASTA** (keys match VDB ids directly).
3. **Override `PLMMSA_SEQUENCE_KEY_FORMAT`** in `.env` to pick a prefix
   that matches whatever id the VDB returns, then populate accordingly.

Until this is decided, the stack will retrieve 0 target sequences from
UniRef50-loaded seq cache for hits returned by the `_test` FAISS.

### Clearing

Targeted (keep tax, drop seq):

```bash
docker compose exec cache-seq redis-cli --scan --pattern 'seq:*' | \
    xargs -n 500 docker compose exec cache-seq redis-cli del
```

Nuke the whole seq + tax store: `docker compose exec cache-seq redis-cli FLUSHDB`.
Wipe on-disk: see "Clearing / wiping" higher up in this file.

## Switching FAISS index size

Step 3 of `PLAN.md`. Each `[vdb.collections.<name>]` block in
`settings.toml` names an `index_path` relative to `VDB_DATA_DIR`. Two
variants typically ship side-by-side under the same collection directory:

| Variant | Filename pattern          | Size    | Loads in             | Use for                    |
| ------- | ------------------------- | ------- | -------------------- | -------------------------- |
| test    | `{collection}_test.faiss` | ~250 MB | seconds, <1 GB RAM   | bring-up / validation / CI |
| full    | `{collection}_vdb.faiss`  | ~90 GB  | minutes, ~100 GB RAM | production MSA generation  |

Swap in place:

```bash
# point both collections at the test variant
sed -i 's/_vdb\.faiss/_test.faiss/g' settings.toml
sed -i 's/_vdb\.faiss_id_mapping/_test.faiss_id_mapping/g' settings.toml
docker compose up -d --no-deps vdb
docker compose exec vdb curl -s http://localhost:8082/health | jq
```

Each collection's `dim` must match the PLM that populated it (Ankh = 1536,
ESM-1b = 1280). `FaissVDB` rejects a dim-mismatched index loudly on load —
check `docker compose logs vdb` if `/health` shows a collection missing.

Sanity check with a synthetic vector (no embedding service needed):

```bash
python3 -c 'import json, random; random.seed(0); print(json.dumps(
    {"collection":"ankh_uniref50",
     "vectors":[[random.gauss(0,1) for _ in range(1536)]],
     "k":5}))' | \
    docker compose exec -T vdb sh -c \
    'curl -sX POST http://localhost:8082/search -H "Content-Type: application/json" -d @-'
```

You should get back five UniRef ids with distances.

## Updating settings

`settings.toml` holds non-secret tunables (per-PLM `enabled`, VDB collection
paths, rate limits, CORS origins, logging knobs). It is bind-mounted
read-only into every service. After edits:

```bash
docker compose restart              # picks up the new file everywhere
# or, narrower
docker compose restart api worker
```

`settings.example.toml` in the repo is authoritative. Use it as the
template when updating `settings.toml`.

## Common failure modes

| Symptom                                                                        | Cause                                     | Fix                                                                                                   |
| ------------------------------------------------------------------------------ | ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `could not select device driver "nvidia"`                                      | `nvidia-container-toolkit` not configured | Install and run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `bind: address already in use` on `api`                                        | Stale local `uvicorn` holding port 8080   | `ss -tlnp \| grep :8080` and kill the offender                                                        |
| Worker logs `Name or service not known` on `embedding:8081`                    | `embedding` container crashed or disabled | `./bin/logs.sh embedding`; verify GPU + HF cache                                                      |
| `/v2/*` returns 401 `E_AUTH_INVALID`                                           | Token revoked or bootstrap rotated        | Re-mint via `/admin/tokens`                                                                           |
| `GET /v2/msa/{id}` says `failed` with `E_INTERNAL "Name or service not known"` | One of embedding / vdb / align is down    | Check that service's health + logs                                                                    |
| Slow `./bin/up.sh` after many restarts                                         | AOF growth on `cache`                     | Wipe `$CACHE_DATA_DIR` (see "Clearing the cache Redis")                                               |

## What lives where

```
.env                     → secrets + host paths (gitignored)
settings.toml            → non-secret tunables (gitignored; copy from settings.example.toml)
$MODEL_CACHE_DIR         → PLM weights + HF cache
$VDB_DATA_DIR            → FAISS index + id-mapping pickles
$CACHE_OPS_DATA_DIR      → Redis persistence: jobs, tokens, rate-limit counters, idempotency keys
$CACHE_SEQ_DATA_DIR      → Redis persistence: UniRef50 seq:* + tax:* lookups
$CACHE_EMB_DATA_DIR      → Redis persistence: PLM embedding cache
docs/                    → this file + cloudflare-tunnel.md + submitting-msa.md
bench/                   → regression harness, not in CI
tests/fixtures/          → CASP15 query fastas + baseline stats, example_a3m.txt
```

## Operator observability

### Structured logs

Every service emits one JSON object per line to stdout. Key loggers:

| Logger            | Emits                                                   |
| ----------------- | ------------------------------------------------------- |
| `plmmsa.access`   | API request summary (method, path, status, duration_ms, client_ip, token_id, request_id) |
| `plmmsa.audit`    | Privileged actions: `msa.submit`, `msa.cancel`, `admin.token.mint`, `admin.token.revoke`, plus token_id / request_id / client_ip |
| `plmmsa.forward`  | Sidecar forward warnings (upstream unreachable, non-JSON) |
| `plmmsa.lifespan` | api shutdown cleanup |

Tail audit only:

```bash
docker compose logs api --since 1h --no-log-prefix | jq -rc 'select(.logger=="plmmsa.audit")'
```

All responses carry `X-Request-ID`. When users report bugs, ask for that
header; every log line the request touched is greppable by it (api +
embedding + vdb + align).

### Prometheus `/metrics`

The api exposes `GET /metrics` in the standard text-exposition format.
It is not rate-limited and not gated behind auth, so scrape from a
trusted network only (or front it with the Cloudflare Access policy for
`/admin/*` — see `docs/cloudflare-tunnel.md`).

Counters / histograms:

- `plmmsa_http_requests_total{method,route,status}`
- `plmmsa_http_request_duration_seconds{method,route}`
- `plmmsa_http_in_flight_requests{method,route}`

Example scrape config:

```yaml
scrape_configs:
  - job_name: plmmsa
    static_configs: [{ targets: ["localhost:8080"] }]
    metrics_path: /metrics
```

### Tuning knobs

Everything below is in `settings.toml` — edit, then `docker compose restart`.

| Section      | Knob                        | Default | Effect                                   |
| ------------ | --------------------------- | ------- | ---------------------------------------- |
| `limits`     | `max_residues_per_chain`    | 1022    | Chain-length cap (edge reject)           |
| `limits`     | `max_chains_paired`         | 16      | Chain-count cap for paired MSAs          |
| `limits`     | `max_body_bytes`            | 10 MB   | Request body cap (ASGI middleware)       |
| `queue`      | `backpressure_threshold`    | 50      | Soft 503 `E_QUEUE_FULL` (Retry-After 5)  |
| `queue`      | `max_queue_depth`           | 200     | Hard 503 `E_QUEUE_FULL` (Retry-After 30) |
| `ratelimit`  | `per_ip_rpm`                | 60      | Per-IP requests/min                      |
| `ratelimit`  | `per_token_rpm`             | 120     | Default per-token requests/min           |
| `api`        | `default_token_ttl_s`       | 90 days | TTL applied when mint omits `expires_at` |
| `logging`    | `level`                     | INFO    | Global log level                         |
| `logging`    | `json_format`               | true    | JSON lines vs. plain formatter           |
| `logging`    | `request_id_header`         | X-Request-ID | Header carrying the request id     |
| `cors`       | `allow_origins`             | localhost + colab | CORS allow-origins             |
| `queue`      | `align_threads`             | 32      | Within-job thread pool for per-target DP fanout. pLM-BLAST ignores this (sequential — see below) |
| `queue`      | `embed_chunk_size`          | 256     | Target-embed batch size per `/embed` call. Length-descending sort means only the first chunk pays max-length padding; 256 fills a 48 GB GPU at `max_length=1022` fp32. Drop on smaller GPUs. |
| `queue`      | `default_k`                 | 1000    | FAISS neighbors per model when the client omits `k`. |
| `aligners.*` | `filter_enabled`            | per-aligner | Apply Algorithm 1 step 5 filter (threshold `min(0.2·len(Q), 8.0)`). Default-on for `plmalign` (dot-product alignment score scale); default-off for `otalign` (transport-mass score scale where the threshold zeroes every hit). Flip per-aligner in settings or per-request via `filter_by_score`. |
| `aligners.otalign` | `fused_sinkhorn`      | false       | Use `sinkhorn_flash.unbalanced_sinkhorn_flash` (torch.compile chunked) instead of the eager torch solver. Adds ~3-10 s warmup cost at server startup; cuts Phase 5 per-target latency on CUDA. Leave off for CPU-only deployments or when debugging. |
| `queue`      | `paired_k_multiplier`       | 3           | Paired-MSA retrieval multiplier. When `paired=true` each chain retrieves `paired_k = multiplier * effective_k` neighbors so the taxonomy-join step still leaves a useful pool per chain. |

### Aligner performance notes

- **PLMAlign** (default). numba-JIT affine-gap SW. ~1 ms per 117×400
  target after warmup. Fans out across `queue.align_threads` — the
  kernel releases the GIL so threads actually parallelize.
- **pLM-BLAST**. Multi-path SW. JIT'd per-cell recurrence is 2-3
  orders of magnitude faster than pure Python, but thread-pool fanout
  **hurts**: a 16-target batch went from 12 s at 1 thread to 170 s at
  16 threads in profiling (suspected numba dispatch-lock contention +
  allocator thrashing). Until fixed, `PlmBlast.align` runs sequentially
  regardless of `align_threads`. For throughput scaling, run multiple
  worker containers (`docker compose up -d --scale worker=N`). For
  latency on a single job, stay on PLMAlign.
- **OTalign**. Unbalanced-Sinkhorn + position-specific-gap DP. Sinkhorn
  + cost matrix run on the torch device declared in
  `aligners.otalign.device` (empty string = auto-detect cuda:0 if
  available, else reads `ANKH_LARGE_DEVICE` from env to match Ankh-
  Large's pin). Requires `align` to have a GPU reservation (already
  set in `docker-compose.yml`). The per-target affine-gap DP is
  numba-JIT'd (`_fill_matrices_jit` in `otalign_dp.py`); sequential
  over targets because each pair shares the same CUDA stream — thread
  fanout doesn't parallelize on GPU.
- **Modes**. All aligners accept `local` / `global`. OTalign
  additionally accepts `glocal`, `q2t`, `t2q` — passed verbatim to the
  DP (no silent `global → glocal` substitution). PLMAlign / pLM-BLAST
  400 on the OTalign-only modes.
- **FlashSinkhorn** (`aligners.otalign.fused_sinkhorn = true`). Wraps
  the Sinkhorn iteration loop in a `torch.compile`'d 16-iteration
  chunk; saves ~3-6 s per OTalign job at k=1000 on the CASP15 targets
  (measured in `docs/submitting-msa.md`'s perf table). The align
  image ships with `gcc/g++/make` specifically so Inductor can build
  the Triton-backed kernels at runtime — if you strip those, warmup
  falls through to the eager solver with a WARN. `_warmup_fused_sinkhorn`
  at server start pays the compile cost once on a 96×256 dummy
  matrix; dynamic shapes mean subsequent job shapes don't recompile.

## Service-to-service wire formats

All large tensor transfers use a compact binary framing
(`plmmsa.align.binary`) instead of JSON, because JSON-encoding ~1 GB of
float32 embeddings dominated per-job latency (measured: 200+ s → 10 s
on a k=1000 OTalign run).

| Endpoint                     | Purpose                                | Wire format |
| ---------------------------- | -------------------------------------- | ----------- |
| `POST /embed`                | Live PLM forward; JSON response.       | JSON in/out (kept for tests + old clients). |
| `POST /embed/bin`            | Same inputs, binary response.          | JSON in / binary out. Orchestrator's default. |
| `POST /embed_by_id`          | ProtT5 shard store; JSON response.     | JSON in/out. |
| `POST /embed_by_id/bin`      | ProtT5 shard store; binary response.   | JSON in / binary out. |
| `POST /align/bin`            | Pairwise alignment request.            | Binary in/out. |

The orchestrator picks binary when `OrchestratorConfig.align_transport
== "binary"` (the default). Tests keep JSON for easy mocking via
`align_transport="json"`.

## ProtT5 shard path index

`ShardStore` uses a SQLite index (`index.db` at the shard root) to map
`.pt` filename → folder (0 / 100 / ... / 900 / missing_embeddings / ...).
Per-request it opens a read-only immutable connection. This works but
the `index.db` lives on `/gpfs` so cold opens are ~tens of ms per
request, and at k=1000 / job cold shard reads still run 30-60 s.

Planned: **Redis-backed path index** (matches upstream DeepFold-PLM's
design: key format like `shard:prott5:<id>` → `<folder>`, populated
once from the sqlite index into `cache-seq` or a dedicated Redis DB).
Expected payoff: path resolution drops from ~30 s → single-digit ms;
remaining Phase-4 cost becomes the actual `torch.load` disk I/O.
