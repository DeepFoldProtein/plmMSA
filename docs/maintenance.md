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
curl -sS $BASE/healthz        # bare liveness (compose + CF edge probe)
curl -sS $BASE/health         # aggregated readiness (fans out to every downstream)
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
- `/healthz` (bare liveness), `/health` (aggregated), `/v2/version`,
  `/metrics`, `/v1/*` (sunset), `/openapi.json`, `/docs`, `/redoc`.

## Starting / stopping the stack

```bash
cp .env.example .env                         # first-time only
cp settings.example.toml settings.toml       # first-time only
./bin/up.sh                                  # full stack (8 services)
./bin/down.sh                                # graceful stop; keeps state volumes
./bin/down.sh -v                             # stop AND delete persistent volumes (destructive)
./bin/logs.sh <service>                      # tail logs, e.g. `./bin/logs.sh worker`
```

`./bin/up.sh` fails fast if `.env` or `settings.toml` are missing. The
eight services that come up on the `plmmsa_net` bridge: `api`,
`embedding`, `vdb`, `align`, `worker`, `cache-ops`, `cache-seq`,
`cache-emb`. To add the public edge, use
`docker compose --profile tunnel up -d`.

## Prerequisites on a fresh host

- Docker ≥ 24 with Compose v2.
- `nvidia-container-toolkit` installed and `nvidia-ctk runtime configure`'d —
  the `embedding` service requests `nvidia` GPU reservations through compose.
  Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
- Writable `MODEL_CACHE_DIR` (PLM weight cache) and three Redis
  persistence dirs (`CACHE_OPS_DATA_DIR`, `CACHE_SEQ_DATA_DIR`,
  `CACHE_EMB_DATA_DIR`). `./bin/up.sh` does not create these — if the
  path in `.env` doesn't exist, compose will try to create it under
  `root`, which is usually wrong. On hosts that pin the Redis container
  uid via `REDIS_CONTAINER_USER` in `.env`, the dirs must also be
  `chown`ed to that uid/gid so Redis can write RDB/AOF files. On
  `deepfold`:
  ```bash
  sudo mkdir -p /gpfs/deepfold/service/cache_{ops,seq,emb}_data
  sudo chown deepfold:deepfold /gpfs/deepfold/service/cache_*_data
  ```

## First-time bring-up — populate caches in this order

The stack will *boot* without any of these populated, but
`POST /v2/msa` will hang on the first ProtT5 fan-out unless every
step here has run at least once. Order matters because the shard
path index lives in `cache-seq`, so populating it before sequences
or after a `cache-seq` wipe is a no-op.

1. **Bring up the persistence + retrieval tier** (no GPU services yet):
   ```bash
   docker compose up -d cache-ops cache-seq cache-emb
   ```
2. **Populate `seq:*` + `tax:*`** in `cache-seq` from the UniRef50
   source (CSV split on `deepfold`, FASTA elsewhere). See
   [§ UniRef50 sequence cache](#uniref50-sequence-cache).
   ~20 min on the deepfold host.
3. **Populate the ProtT5 shard path index** in the same `cache-seq`
   instance (DB 0). See [§ ProtT5 shard path index](#prott5-shard-path-index).
   ~55 min on the deepfold host. Skipping this step is the most common
   cause of `/embed_by_id/bin` timeouts and `pipeline failed: ReadTimeout`
   in worker logs — the orchestrator falls back to live `/embed`, which
   on a 1000-target job is also slow enough to hit the 900 s cap.
4. **Pre-warm PLM weights** under `MODEL_CACHE_DIR` so the first
   `embedding` startup doesn't try to fetch from HuggingFace at
   runtime (the container runs with `HF_HUB_OFFLINE=1` in production).
   See [§ Populating model weights](#populating-model-weights).
5. **Bring up the GPU + API tier:**
   ```bash
   ./bin/up.sh
   curl -fsS http://localhost:8080/health | jq '.status'   # → "ok"
   ```
6. **Smoke a job** end-to-end against a CASP15 fixture to confirm
   every wire works:
   ```bash
   SEQ=$(tail -n +2 tests/fixtures/casp15/T1120.fasta | tr -d '\n')
   JID=$(curl -sS -X POST http://localhost:8080/v2/msa \
       -H 'Content-Type: application/json' \
       -d "$(jq -Rn --arg s "$SEQ" '{sequences:[$s],query_ids:["T1120"],k:50}')" \
       | jq -r .job_id)
   while [[ "$(curl -sS http://localhost:8080/v2/msa/$JID | jq -r .status)" != "succeeded" ]]; do sleep 2; done
   curl -sS http://localhost:8080/v2/msa/$JID | jq '.result.stats'
   ```
   Expect end-to-end ≤ 5 s on a warm stack with `k=50`,
   `stats.depth ≈ 90`, no `cache_hit` field on first run.

## Controlling where state lives

`.env` keys that determine on-host paths:

| Key                    | Default            | Used by                                                                                                        |
| ---------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| `MODEL_CACHE_DIR`      | `./model_cache`    | `embedding` — PLM weights (HuggingFace cache)                                                                  |
| `VDB_DATA_DIR`         | `./vdb_data`       | `vdb` — FAISS index + id-mapping pickles                                                                       |
| `CACHE_OPS_DATA_DIR`   | `./cache_ops_data` | `cache-ops` — jobs, tokens, rate limits (AOF)                                                                  |
| `CACHE_SEQ_DATA_DIR`   | `./cache_seq_data` | `cache-seq` — `seq:*` + `tax:*` lookups + shard path index (RDB)                                               |
| `CACHE_EMB_DATA_DIR`   | `./cache_emb_data` | `cache-emb` — completed-MSA result cache (RDB + LRU)                                                           |
| `REDIS_CONTAINER_USER` | unset (uid 999)    | all three Redis services — set `uid:gid` to own RDB/AOF files on disk. Example on `deepfold`: `220104:220104`. |

Point these at `/gpfs` for multi-machine persistence; keep them local
for a single-host dev deployment. On the `deepfold` host the
authoritative paths are:

- `MODEL_CACHE_DIR=/gpfs/deepfold/model_cache` (HF `hub/` layout,
  shared read-only across containers; do **not** use the dead
  `/store/deepfold/huggingface`).
- `CACHE_*_DATA_DIR=/gpfs/deepfold/service/cache_{ops,seq,emb}_data`
  owned `deepfold:deepfold`, with `REDIS_CONTAINER_USER=220104:220104`
  so Redis writes as the host user.

## Three Redis instances, one per role

| Instance    | Image            | Holds                                                                           | Persistence    | Eviction                                           | Typical size                    |
| ----------- | ---------------- | ------------------------------------------------------------------------------- | -------------- | -------------------------------------------------- | ------------------------------- |
| `cache-ops` | redis:7.4-alpine | `plmmsa:job:*`, `plmmsa:queue`, `admintoken:*`, idempotency keys, rate counters | AOF on, no RDB | `noeviction`                                       | MBs                             |
| `cache-seq` | redis:7.4-alpine | `seq:*`, `tax:*`, `shard:<model>:<id>` (UniRef50 + taxonomy + shard index)      | RDB snapshots  | `noeviction` (configurable via `CACHE_SEQ_POLICY`) | ~30 GB                          |
| `cache-emb` | redis:7.4-alpine | `plmmsa:result:*` — completed-MSA result cache, keyed by canonical submit hash  | RDB snapshots  | `allkeys-lru`                                      | capped by `CACHE_EMB_MAXMEMORY` |

Why three instead of one:

- **Eviction.** `cache-emb` can evict safely (results are rebuildable
  by re-running the pipeline); `cache-ops` must never evict (you'd lose
  a queued job or revoke a token by accident). A single Redis can't
  have two policies at once.
- **Restart time.** `cache-ops` reloads in a second and the API is live;
  `cache-seq` takes minutes to replay the UniRef50 RDB but nothing
  user-facing depends on it at that moment.
- **Blast radius.** A bug that FLUSHDBs the wrong instance is less bad
  when roles are isolated.

### Result cache (`plmmsa:result:*` on `cache-emb`)

Keyed by `sha256` of a canonicalized submit body (sequences uppercased
+ whitespace-stripped, chain order preserved, non-output-affecting
fields like `force_recompute` and `request_id` dropped; see
[`src/plmmsa/jobs/result_cache.py`](../src/plmmsa/jobs/result_cache.py)).

- **On submit**: `api` computes the key and, on hit, synthesizes a
  `succeeded` Job record and returns its id without enqueueing.
  `result.stats.cache_hit = true` marks served-from-cache responses so
  downstream observability can distinguish them from fresh compute.
- **On worker success**: the worker writes the canonical result into
  `cache-emb` with TTL from `PLMMSA_RESULT_CACHE_TTL_S` (default 30
  days). Failures to write never fail the job — the MSA is already
  persisted via `mark_succeeded`.
- **Opt-out per request**: `{"force_recompute": true}` on
  `POST /v2/msa` bypasses the cache and still overwrites on completion.
- **Cache keyspace bump**: `CACHE_VERSION = "v1"` — bump inside
  `result_cache.py` when the canonicalization changes to invalidate
  every prior entry.

Disable globally by unsetting `PLMMSA_RESULT_CACHE_URL` in `.env`;
`ResultCache(None)` is a no-op that keeps the rest of the pipeline
working unchanged.

## Clearing / wiping

Per-role, non-destructive:

```bash
# Drop just the sequence cache (keep queue + tokens + result cache).
docker compose exec cache-seq redis-cli --scan --pattern 'seq:*' | \
    xargs -n 500 docker compose exec cache-seq redis-cli del

# Drop just the result cache (forces every submit to re-run the pipeline).
docker compose exec cache-emb redis-cli FLUSHDB

# Drop only the idempotency keys on cache-ops (preserves jobs + tokens
# + rate counters). Resubmits with identical payloads will create new
# jobs instead of returning the prior job_id.
docker compose exec cache-ops sh -lc \
    'redis-cli --scan --pattern "idem:*" | xargs -r redis-cli DEL'
```

**Use both together after a wire-format-changing rollout.** When a
deploy changes the canonical submit shape, the score scale, or the
emitted A3M format, the result cache holds entries that are no longer
on the new contract — and the idempotency keys (10-min TTL) point
resubmits at the old jobs. Run both clears after `docker compose up
-d --build` so fresh submissions re-enter the queue cleanly:

```bash
docker exec plmmsa-cache-emb-1 redis-cli FLUSHDB
docker exec plmmsa-cache-ops-1 sh -lc \
    'redis-cli --scan --pattern "idem:*" | xargs -r redis-cli DEL'
```

(The `docker exec` form vs. `docker compose exec` works the same;
the former is what shows up in shell history when troubleshooting on
the host without `cd`-ing into the project. Both reach the same
container.)

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

Two sources are authoritative — pick one:

1. **UniRef50 CSV split** (preferred on `deepfold` — matches the
   `.pt` shard layout used by the ProtT5 shard store):
   ```
   /gpfs/database/milvus/datasets/uniref50_t5/split/*.csv   # ~6,585 files, ~27 GB total
   ```
   Columns: `accession, description, sequence, length, length_group`.
   The full UniRef50 cluster id (`UniRef50_<accession>`) and
   `TaxID=<n>` both live inside `description`.

2. **UniRef50 FASTA** (fallback for hosts without the CSV split):
   ```
   /gpfs/database/casp16/uniref50/uniref50.fasta   # 26 GB, ~60 M sequences
   ```

### Quick subset validation (seconds)

```bash
# CSV: point at a single shard file
docker compose up -d cache-seq worker
docker compose run --rm \
    -v /gpfs/database/milvus/datasets/uniref50_t5/split:/csvs:ro \
    worker uv run python -m plmmsa.tools.build_sequence_cache \
    --csv /csvs/0-1000.csv \
    --redis-url redis://cache-seq:6379 \
    --key-format 'seq:UniRef50_{id}' \
    --tax-key-format 'tax:UniRef50_{id}' \
    --batch 500

# FASTA: a 50k-record head of the full file
head -50000 /gpfs/database/casp16/uniref50/uniref50.fasta > /tmp/uniref50_head.fasta
docker compose cp /tmp/uniref50_head.fasta worker:/tmp/uniref50_head.fasta
docker compose exec -T worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /tmp/uniref50_head.fasta --redis-url redis://cache-seq:6379 --batch 200

# Spot-check a few ids for both seq and tax.
for id in $(docker compose exec -T cache-seq redis-cli --scan --pattern 'seq:UniRef50_*' | head -3); do
    name="${id#seq:}"
    echo -n "$name  seq len: "; docker compose exec -T cache-seq redis-cli STRLEN "seq:$name"
    echo -n "$name  tax id:  "; docker compose exec -T cache-seq redis-cli GET "tax:$name"
done
```

### Full load

From the CSV split (authoritative on `deepfold`):

```bash
docker compose up -d cache-seq worker
docker compose run --rm \
    -v /gpfs/database/milvus/datasets/uniref50_t5/split:/csvs:ro \
    worker uv run python -m plmmsa.tools.build_sequence_cache \
    --csv-dir /csvs \
    --redis-url redis://cache-seq:6379 \
    --key-format 'seq:UniRef50_{id}' \
    --tax-key-format 'tax:UniRef50_{id}' \
    --batch 5000
```

Files are lex-sorted and processed in order. If a run is interrupted,
resume via `--start <idx>` / `--stop <idx>` against the same sort
order; the INFO log emits the current file index on every boundary.

From the FASTA (fallback):

```bash
docker compose run --rm \
    -v /gpfs/database/casp16/uniref50:/uniref-src:ro \
    worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /uniref-src/uniref50.fasta \
    --redis-url redis://cache-seq:6379 \
    --batch 5000
```

Every record emits a `seq:*` and (when the source has `TaxID=<n>`) a
`tax:*` key, so final DBSIZE is ~120 M at ~30 GB resident. Matches the
`CACHE_SEQ_MAXMEMORY=0` / `noeviction` policy in `.env.example` —
growth beyond that fails writes loudly rather than silently evicting
loaded records.

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

| Symptom                                                                                                                                          | Cause                                                                                                                           | Fix                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `could not select device driver "nvidia"`                                                                                                        | `nvidia-container-toolkit` not configured                                                                                       | Install and run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`                                                                                                        |
| `bind: address already in use` on `api`                                                                                                          | Stale local `uvicorn` holding port 8080                                                                                         | `ss -tlnp \| grep :8080` and kill the offender                                                                                                                                                               |
| Worker logs `Name or service not known` on `embedding:8081`                                                                                      | `embedding` container crashed or disabled                                                                                       | `./bin/logs.sh embedding`; verify GPU + HF cache                                                                                                                                                             |
| `/v2/*` returns 401 `E_AUTH_INVALID`                                                                                                             | Token revoked or bootstrap rotated                                                                                              | Re-mint via `/admin/tokens`                                                                                                                                                                                  |
| `GET /v2/msa/{id}` says `failed` with `E_INTERNAL "Name or service not known"`                                                                   | One of embedding / vdb / align is down                                                                                          | Check that service's health + logs                                                                                                                                                                           |
| Slow `./bin/up.sh` after many restarts                                                                                                           | AOF/RDB growth on one of the three caches                                                                                       | Wipe the offender's `$CACHE_*_DATA_DIR` (see "Clearing / wiping")                                                                                                                                            |
| Redis container logs `Permission denied` writing `/data/dump.rdb`                                                                                | Host dir owner mismatches `REDIS_CONTAINER_USER`                                                                                | `chown` the host dir to the configured uid/gid, or unset `REDIS_CONTAINER_USER` to fall back to image default 999.                                                                                           |
| Worker logs `shard lookup failed for model=prott5; falling back to /embed` then `httpx.ReadTimeout` after 900 s, job ends `failed E_INTERNAL`    | `shard:prott5:*` index empty in `cache-seq` (sqlite + dir-scan fallback can't beat the http timeout on `/gpfs`)                 | Run `python -m plmmsa.tools.build_shard_index` once — see [§ ProtT5 shard path index](#prott5-shard-path-index). Re-run after any `cache-seq` wipe.                                                          |
| `embedding` container flips to `(unhealthy)` while jobs are running but `/embed/bin` requests still succeed                                      | /health probe queues behind a long GPU-bound `/embed/bin` and trips the healthcheck `timeout` (5 s default was too tight)       | Bumped to 30 s in `docker-compose.yml`; if you tune it down, watch out under `k≥1000` Ankh-Large jobs                                                                                                        |
| `prometheus` container crashloops with `panic: Unable to create mmap-ed active query log` / `open /prometheus/queries.active: permission denied` | Compose's `create_host_path: true` made `$PROMETHEUS_DATA_DIR` root-owned; container runs as uid 65534 (nobody) and can't write | Stop prometheus, `rm -rf` the empty data dir, recreate it owned by your host user, set `PROMETHEUS_CONTAINER_USER=$(id -u):$(id -g)` in `.env`, and recreate the container. See `.env.example` for the knob. |

## What lives where

```
.env                     → secrets + host paths (gitignored)
settings.toml            → non-secret tunables (gitignored; copy from settings.example.toml)
$MODEL_CACHE_DIR         → PLM weights + HF cache
$VDB_DATA_DIR            → FAISS index + id-mapping pickles
$CACHE_OPS_DATA_DIR      → Redis persistence: jobs, tokens, rate-limit counters, idempotency keys
$CACHE_SEQ_DATA_DIR      → Redis persistence: UniRef50 seq:* + tax:* lookups
$CACHE_EMB_DATA_DIR      → Redis persistence: completed-MSA result cache (plmmsa:result:*)
docs/                    → this file + cloudflare-tunnel.md + submitting-msa.md
bench/                   → regression harness, not in CI
tests/fixtures/          → CASP15 query fastas + baseline stats, example_a3m.txt
```

## Operator observability

### Structured logs

Every service emits one JSON object per line to stdout. Key loggers:

| Logger                                 | Emits                                                                                                                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plmmsa.access`                        | api request summary (method, path, status, duration_ms, client_ip, token_id, request_id)                                                                                          |
| `plmmsa.access.embedding`              | sidecar request summary (method, path, status, duration_ms, request_id). Same shape for `.vdb` / `.align`.                                                                        |
| `plmmsa.audit`                         | Privileged actions: `msa.submit`, `msa.submit.cache_hit`, `msa.submit.idempotent`, `msa.cancel`, `admin.token.mint`, `admin.token.revoke`, plus token_id / request_id / client_ip |
| `plmmsa.forward`                       | api → sidecar forward warnings (upstream unreachable, non-JSON)                                                                                                                   |
| `plmmsa.api.error`                     | PlmMSAError returned from api routes: 5xx with full traceback, 4xx as warning with method / path / code / message.                                                                |
| `plmmsa.embedding` / `.vdb` / `.align` | Per-sidecar PlmMSAError responses, same 5xx-with-traceback / 4xx-warning policy as api.                                                                                           |
| `plmmsa.lifespan`                      | api shutdown cleanup                                                                                                                                                              |

Tail audit only:

```bash
docker compose logs api --since 1h --no-log-prefix | jq -rc 'select(.logger=="plmmsa.audit")'
```

Follow one submission across services by `request_id`:

```bash
RID=edge-trace-id
docker compose logs api embedding vdb align worker --since 15m --no-log-prefix \
    | jq -rc --arg rid "$RID" 'select(.request_id == $rid)'
```

All responses carry `X-Request-ID`. Clients that supply one on the
way in have it echoed back and threaded onto every downstream call
(api → embedding / vdb / align) and onto the worker's orchestrator
calls for that job. When users report bugs, ask for that header.

### Prometheus `/metrics`

Every service exposes `/metrics` in the standard text-exposition
format. The endpoints are not rate-limited and not gated behind auth —
scrape from a trusted network only (or front `/admin/*` + metrics with
the Cloudflare Access policy — see `docs/cloudflare-tunnel.md`).

| Service   | Scrape target            | Notes                                                                                                                     |
| --------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| api       | `api:8080/metrics`       | Published to the host on `${API_HOST_PORT}` (default 8080).                                                               |
| embedding | `embedding:8081/metrics` | Bridge-only. Scrape from a container on `plmmsa_net`.                                                                     |
| vdb       | `vdb:8082/metrics`       | Bridge-only.                                                                                                              |
| align     | `align:8083/metrics`     | Bridge-only.                                                                                                              |
| worker    | `worker:9090/metrics`    | Embedded `prometheus_client.start_http_server`. Port pinned via `PLMMSA_WORKER_METRICS_PORT` (default 9090, bridge-only). |

HTTP counters / histograms (emitted by api / embedding / vdb / align):

- `plmmsa_http_requests_total{service,method,route,status}`
- `plmmsa_http_request_duration_seconds{service,method,route}`
- `plmmsa_http_in_flight_requests{service,method,route}`

Worker-specific metrics (emitted by worker):

- `plmmsa_worker_jobs_processed_total{status}` — counter, status ∈
  `{succeeded, failed, cancelled}`.
- `plmmsa_worker_pipeline_duration_seconds` — histogram of orchestrator
  wall time per job; buckets span 1 s → 30 min.
- `plmmsa_worker_queue_depth` — gauge, sampled every 5 s from
  `LLEN plmmsa:queue`.

#### Bundled Prometheus + GPU exporter (`observability` compose profile)

Two services live behind the same off-by-default profile:

- `prometheus` — TSDB + scrape engine + web UI.
- `dcgm-exporter` — NVIDIA GPU telemetry (~50 metrics named
  `DCGM_FI_DEV_*`: utilization, memory, temperature, power, ECC,
  NVLink, etc.). Joins `plmmsa_net`; reserves all GPUs via the same
  nvidia driver the embedding service uses.

Bring both up at once:

```bash
docker compose --profile observability up -d prometheus dcgm-exporter
```

Then open `http://localhost:${PROMETHEUS_HOST_PORT:-9091}` (default
9091; the container's internal port stays at 9090 to avoid clashing
with the worker's embedded scrape target). Scrape config lives at
[`services/prometheus/prometheus.yml`](../services/prometheus/prometheus.yml);
TSDB persistence + retention are knob'd via `PROMETHEUS_DATA_DIR` /
`PROMETHEUS_RETENTION` in `.env` (see `.env.example`).

Tear down without losing TSDB:

```bash
docker compose --profile observability stop prometheus
```

#### Navigating the Prometheus web UI

Default URL: **`http://localhost:${PROMETHEUS_HOST_PORT:-9091}`**.
Useful pages once it's open:

| URL                       | What it shows                                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/` (Graph)               | Type a PromQL query into the box; the **Table** tab shows current values, the **Graph** tab plots over time. Bookmark queries via the URL bar — Prometheus encodes the query into the URL.        |
| `/targets`                | Live scrape health for every target. Click a target to see its raw `/metrics` blob. The "Last Scrape", "Last Error", and "Scrape Duration" columns are the first stop when something looks stale. |
| `/service-discovery`      | What targets the SD config resolved to before relabeling — useful when you change `prometheus.yml` and want to confirm what hit the scrape pool.                                                  |
| `/rules`, `/alerts`       | Recording / alerting rules (none configured by default in this stack).                                                                                                                            |
| `/tsdb-status`            | TSDB health: head series count, head chunk count, retention. Use it to gauge how big the on-disk data dir is going to get.                                                                        |
| `/flags`                  | Effective command-line flags. Confirms `--storage.tsdb.retention.time` etc. landed.                                                                                                               |
| `/status/runtime`         | Goroutine / GC / start time / uptime / memory.                                                                                                                                                    |
| `/api/v1/query?query=...` | Raw HTTP API. The Graph UI is just a thin shell over this — copy a working PromQL into curl/jq for scripting.                                                                                     |

**Workflow tip — autocompletion.** The query box has incremental
metric-name autocomplete. Type `plmmsa_` to see every plmMSA metric
the stack emits; type `DCGM_FI_DEV_` for GPU metrics. Hit
**"Use Query"** on a row in `/targets` to pre-fill the Graph page
with a per-target `up` query for that scrape.

#### Alerting rules + the `/alerts` page

The bundled Prometheus loads
[`services/prometheus/alerts.yml`](../services/prometheus/alerts.yml)
as a `rule_files` entry. Ten rules across five groups cover the
operational fault classes we've actually hit on this stack:

| group             | rule                 | severity | what it catches                                               |
| ----------------- | -------------------- | -------- | ------------------------------------------------------------- |
| `plmmsa-uptime`   | `ScrapeTargetDown`   | critical | scrape target down 2m+                                        |
| `plmmsa-errors`   | `ApiServerErrorRate` | warning  | 5xx rate > 0 sustained                                        |
| `plmmsa-errors`   | `WorkerJobFailures`  | warning  | worker `mark_failed` rate sustained                           |
| `plmmsa-latency`  | `PipelineP95High`    | warning  | worker pipeline p95 > 90 s for 10 m (regression / cold cache) |
| `plmmsa-latency`  | `EmbeddingP95High`   | warning  | `/embed/bin` p95 > 30 s (GPU pressure)                        |
| `plmmsa-capacity` | `QueueBackpressure`  | warning  | queue depth > 30 (heading for `503 E_QUEUE_FULL`)             |
| `plmmsa-capacity` | `QueueStuck`         | critical | queue > 0 + zero successes in 10 m (worker hung)              |
| `plmmsa-gpu`      | `GpuEccErrors`       | critical | DCGM SBE/DBE ECC counter > 0 (hardware fire alarm)            |
| `plmmsa-gpu`      | `GpuTempHigh`        | warning  | sustained > 80 °C (RTX 6000 Ada throttles ~87 °C)             |
| `plmmsa-gpu`      | `GpuMemoryNearFull`  | warning  | `< 2 GiB` free for 2 m (predict OOM on next long /embed)      |

View pending + firing at **`http://localhost:${PROMETHEUS_HOST_PORT:-9091}/alerts`**.

> **No notifications by default.** Without an Alertmanager configured
> in `prometheus.yml`'s `alerting:` block, alerts only surface in the
> Prometheus UI and via the `ALERTS{...}` time series — they don't
> page anyone. To add notifications, ship an `alertmanager` container
> alongside this stack and uncomment the `alerting:` block in
> [`services/prometheus/prometheus.yml`](../services/prometheus/prometheus.yml).

After editing alerts:

```bash
docker compose --profile observability up -d --force-recreate prometheus
```

(restart on its own won't pick up new bind mounts; force-recreate or
add `--web.enable-lifecycle` and POST to `/-/reload`).

#### Useful starter queries

A complete copy-paste cookbook (sectioned by the operational
question each query answers — health, latency, GPU, capacity,
troubleshooting) lives at
[`services/prometheus/queries.md`](../services/prometheus/queries.md).
A few starters from there:

```promql
# Per-service request rate (req/s, 1m window).
sum by (service) (rate(plmmsa_http_requests_total[1m]))

# api p95 latency by route.
histogram_quantile(0.95, sum by (le, route) (rate(plmmsa_http_request_duration_seconds_bucket{service="api"}[5m])))

# Worker pipeline p95.
histogram_quantile(0.95, rate(plmmsa_worker_pipeline_duration_seconds_bucket[10m]))

# Job throughput by terminal status (succeeded / failed / cancelled).
rate(plmmsa_worker_jobs_processed_total[5m])

# Live queue depth (sampled every 5 s by the worker).
plmmsa_worker_queue_depth

# 5xx rate per service — first place to look if something feels off.
sum by (service) (rate(plmmsa_http_requests_total{status=~"5.."}[5m]))

# Are all targets up?
up

# GPU utilisation per card (%).
DCGM_FI_DEV_GPU_UTIL

# GPU memory used / free (MiB), per card.
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_FB_FREE

# Temperature (°C) and power (W) per card.
DCGM_FI_DEV_GPU_TEMP
DCGM_FI_DEV_POWER_USAGE

# ECC errors — should always be 0; non-zero = report to ops.
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL

# Cross-correlation: embedding latency vs GPU 0 utilisation in the
# same window — useful to confirm that p95 spikes coincide with GPU
# contention (rather than e.g. Redis latency).
histogram_quantile(0.95,
  sum by (le) (rate(plmmsa_http_request_duration_seconds_bucket{service="embedding"}[5m])))
DCGM_FI_DEV_GPU_UTIL{gpu="0"}
```

#### External Prometheus (no bundled instance)

If you already run a Prometheus elsewhere, the same scrape config
works as long as the scraper can reach the bridge:

```yaml
scrape_configs:
  - job_name: plmmsa
    static_configs:
      - targets:
          - api:8080
          - embedding:8081
          - vdb:8082
          - align:8083
          - worker:9090
    metrics_path: /metrics
```

If you only scrape from outside the bridge (e.g. a host-level
Prometheus), publish additional host ports on `docker-compose.yml` as
needed. The api port is published by default; the rest are bridge-only
to avoid accidental exposure of internal telemetry.

### Tuning knobs

Everything below is in `settings.toml` — edit, then `docker compose restart`.

| Section            | Knob                     | Default           | Effect                                                                                                                                                                                                                                                                                             |
| ------------------ | ------------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `limits`           | `max_residues_per_chain` | 1022              | Chain-length cap (edge reject)                                                                                                                                                                                                                                                                     |
| `limits`           | `max_chains_paired`      | 16                | Chain-count cap for paired MSAs                                                                                                                                                                                                                                                                    |
| `limits`           | `max_body_bytes`         | 10 MB             | Request body cap (ASGI middleware)                                                                                                                                                                                                                                                                 |
| `queue`            | `backpressure_threshold` | 50                | Soft 503 `E_QUEUE_FULL` (Retry-After 5)                                                                                                                                                                                                                                                            |
| `queue`            | `max_queue_depth`        | 200               | Hard 503 `E_QUEUE_FULL` (Retry-After 30)                                                                                                                                                                                                                                                           |
| `ratelimit`        | `per_ip_rpm`             | 60                | Per-IP requests/min                                                                                                                                                                                                                                                                                |
| `ratelimit`        | `per_token_rpm`          | 120               | Default per-token requests/min                                                                                                                                                                                                                                                                     |
| `api`              | `default_token_ttl_s`    | 90 days           | TTL applied when mint omits `expires_at`                                                                                                                                                                                                                                                           |
| `logging`          | `level`                  | INFO              | Global log level                                                                                                                                                                                                                                                                                   |
| `logging`          | `json_format`            | true              | JSON lines vs. plain formatter                                                                                                                                                                                                                                                                     |
| `logging`          | `request_id_header`      | X-Request-ID      | Header carrying the request id                                                                                                                                                                                                                                                                     |
| `cors`             | `allow_origins`          | localhost + colab | CORS allow-origins                                                                                                                                                                                                                                                                                 |
| `queue`            | `align_threads`          | 32                | Within-job thread pool for per-target DP fanout. pLM-BLAST ignores this (sequential — see below)                                                                                                                                                                                                   |
| `queue`            | `embed_chunk_size`       | 256               | Target-embed batch size per `/embed` call. Length-descending sort means only the first chunk pays max-length padding; 256 fills a 48 GB GPU at `max_length=1022` fp32. Drop on smaller GPUs.                                                                                                       |
| `queue`            | `default_k`              | 1000              | FAISS neighbors per model when the client omits `k`.                                                                                                                                                                                                                                               |
| `aligners.*`       | `filter_enabled`         | per-aligner       | Apply Algorithm 1 step 5 filter (threshold `min(0.2·len(Q), 8.0)`). Default-on for `plmalign` (dot-product alignment score scale); default-off for `otalign` (transport-mass score scale where the threshold zeroes every hit). Flip per-aligner in settings or per-request via `filter_by_score`. |
| `aligners.otalign` | `fused_sinkhorn`         | false             | Use `sinkhorn_flash.unbalanced_sinkhorn_flash` (torch.compile chunked) instead of the eager torch solver. Adds ~3-10 s warmup cost at server startup; cuts Phase 5 per-target latency on CUDA. Leave off for CPU-only deployments or when debugging.                                               |
| `queue`            | `paired_k_multiplier`    | 3                 | Paired-MSA retrieval multiplier. When `paired=true` each chain retrieves `paired_k = multiplier * effective_k` neighbors so the taxonomy-join step still leaves a useful pool per chain.                                                                                                           |

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

| Endpoint                | Purpose                              | Wire format                                   |
| ----------------------- | ------------------------------------ | --------------------------------------------- |
| `POST /embed`           | Live PLM forward; JSON response.     | JSON in/out (kept for tests + old clients).   |
| `POST /embed/bin`       | Same inputs, binary response.        | JSON in / binary out. Orchestrator's default. |
| `POST /embed_by_id`     | ProtT5 shard store; JSON response.   | JSON in/out.                                  |
| `POST /embed_by_id/bin` | ProtT5 shard store; binary response. | JSON in / binary out.                         |
| `POST /align/bin`       | Pairwise alignment request.          | Binary in/out.                                |

The orchestrator picks binary when `OrchestratorConfig.align_transport
== "binary"` (the default). Tests keep JSON for easy mocking via
`align_transport="json"`.

## ProtT5 shard path index

`ShardStore` resolves `<id>.pt` → folder (0 / 100 / 200 / ... / 1000)
through a Redis MGET against `cache-seq` DB 0 (key format
`shard:prott5:<bare_id>` → `<folder>`). The fallback is a sqlite
index (`index.db` at the shard root) for ids Redis doesn't know,
followed by a per-shard-folder filesystem scan.

**The Redis path index must be populated** — without it, every
`/embed_by_id/bin` request falls back to sqlite + filesystem scan on
`/gpfs`, which on a cold cache dominates phase 4 (~30-60 s on a
k=1000 job; on the rebased stack a 50-target T1120 hung past the 900 s
http timeout because the sqlite + dir scan did not finish in time).
The compose default sets `PLMMSA_SHARD_INDEX_REDIS_URL=redis://cache-seq:6379/0`
on both `embedding` and `worker`, so the only step left is to actually
write the keys.

### Populate

Run once per shard-store rebuild:

```bash
docker compose up -d cache-seq worker
docker compose run --rm --no-deps \
    -v /gpfs/database/milvus/datasets/uniref50_t5/datasets:/shards:ro \
    worker uv run python -m plmmsa.tools.build_shard_index \
    --sqlite /shards/index.db \
    --redis-url redis://cache-seq:6379/0 \
    --model prott5 \
    --batch 10000
```

On the `deepfold` host this writes ~64.7 M `shard:prott5:*` keys at
~20 k/s — about 55 minutes on a populated `cache-seq`. Progress
prints every 100 k keys; the final line is
`shard_index: done, N keys written`.

After it lands, `cache-seq` DBSIZE should jump by ~64.7 M (added on
top of the ~131.6 M `seq:*`+`tax:*` keys from the
`build_sequence_cache` step above), landing around 196 M total.

### Verify

```bash
docker compose exec cache-seq redis-cli -n 0 GET shard:prott5:A0A009QGN5
# → "0"   (the folder under uniref50_t5/datasets/ that holds .pt)
```

Run a 50-target T1120 submission (recipe in [submitting-msa.md](./submitting-msa.md))
and confirm the worker log shows neither `shard lookup failed` nor
`httpx.ReadTimeout` — phase 4 should land in well under a second.

### Repopulate after a cache-seq wipe

`shard:prott5:*` lives in `cache-seq`, so a `FLUSHDB` or
`$CACHE_SEQ_DATA_DIR` wipe takes the shard index with it. Re-run
`build_shard_index` after any rebuild that touches `cache-seq`.
