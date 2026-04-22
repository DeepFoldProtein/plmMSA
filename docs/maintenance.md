# plmMSA maintenance

Runbook for keeping the stack healthy on the `deepfold` host. Pair with
`docs/cloudflare-tunnel.md` for the public-edge piece.

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

| Key               | Default         | Used by                                       |
| ----------------- | --------------- | --------------------------------------------- |
| `MODEL_CACHE_DIR` | `./model_cache` | `embedding` — PLM weights (HuggingFace cache) |
| `VDB_DATA_DIR`    | `./vdb_data`    | `vdb` — FAISS index + id-mapping pickles      |
| `CACHE_DATA_DIR`  | `./cache_data`  | `cache` — Redis persistence (AOF / RDB)       |

Point these at `/gpfs` or similar for multi-machine persistence; keep them
local for a single-host dev deployment.

## Clearing the cache Redis

The `cache` Redis holds three kinds of data:

- **Job records + queue** — `plmmsa:job:*`, `plmmsa:queue`.
- **Admin tokens** — `admintoken:rec:*`, `admintoken:hash:*`, `admintoken:all`.
- **Sequence lookup** (populated by `plmmsa.tools.build_sequence_cache`) —
  `seq:*` by default.

The sequence store is by far the largest after a UniRef50 load. Three ways
to reclaim disk when it's no longer useful, in increasing severity:

```bash
# 1. Drop only the sequence cache, keep queue + tokens.
docker compose exec cache redis-cli --scan --pattern 'seq:*' | \
    xargs -n 500 docker compose exec cache redis-cli del

# 2. Flush the whole Redis database. Queue + tokens + seq cache all gone.
docker compose exec cache redis-cli FLUSHDB

# 3. Full wipe: stop compose, delete the host directory, start fresh. This
#    is the fastest way to bring Redis up with a clean, small on-disk file
#    after many writes; the AOF bloats over time and compaction is slow.
./bin/down.sh
rm -rf "$(grep '^CACHE_DATA_DIR=' .env | cut -d= -f2-)/*"
./bin/up.sh
```

**Why "clear cache_data for fast loading" matters:** Redis AOF replay on
startup can take minutes for a heavily-written UniRef50 cache. If you only
need the cache for a short-lived benchmark and don't need persistence across
restarts, delete `CACHE_DATA_DIR/*` between runs — Redis comes up cold and
fast. The sequence cache is trivially rebuildable from the source FASTA:

```bash
uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /path/to/uniref50.fasta \
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
from Redis keyed as `seq:{id}`. Populate once per host; the keys live on
`$CACHE_DATA_DIR` and survive `./bin/down.sh` / `up.sh`.

### On the deepfold host

The UniRef50 FASTA lives at:

```
/gpfs/database/casp16/uniref50/uniref50.fasta   # 26 GB, ~60 M sequences
```

### Quick subset validation (seconds)

```bash
head -2000 /gpfs/database/casp16/uniref50/uniref50.fasta > /tmp/uniref50_head.fasta
docker compose up -d cache worker
docker compose cp /tmp/uniref50_head.fasta worker:/tmp/uniref50_head.fasta
docker compose exec -T worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /tmp/uniref50_head.fasta --redis-url redis://cache:6379 --batch 200
docker compose exec -T cache redis-cli --scan --pattern 'seq:UniRef50_*' | head
```

### Full load (~60 minutes)

```bash
docker compose up -d cache worker
docker compose exec -T worker uv run python -m plmmsa.tools.build_sequence_cache \
    --fasta /gpfs/database/casp16/uniref50/uniref50.fasta \
    --redis-url redis://cache:6379 \
    --batch 5000
```

Progress is logged every 10 batches. Redis AOF grows to roughly the same
size as the FASTA. Disk usage stabilizes once the load finishes.

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

Targeted:

```bash
docker compose exec cache redis-cli --scan --pattern 'seq:*' | \
    xargs -n 500 docker compose exec cache redis-cli del
```

Nuclear: see "Clearing the cache Redis" higher up in this file.

## Switching FAISS index size

Step 3 of `PLAN.md`. Each `[vdb.collections.<name>]` block in
`settings.toml` names an `index_path` relative to `VDB_DATA_DIR`. Two
variants typically ship side-by-side under the same collection directory:

| Variant  | Filename pattern              | Size     | Loads in             | Use for                           |
| -------- | ----------------------------- | -------- | -------------------- | --------------------------------- |
| test     | `{collection}_test.faiss`     | ~250 MB  | seconds, <1 GB RAM   | bring-up / validation / CI        |
| full     | `{collection}_vdb.faiss`      | ~90 GB   | minutes, ~100 GB RAM | production MSA generation         |

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
$CACHE_DATA_DIR          → Redis persistence: jobs, tokens, seq:* cache
docs/                    → this file + cloudflare-tunnel.md
bench/                   → regression harness, not in CI
tests/fixtures/          → CASP15 query fastas + baseline stats, example_a3m.txt
```
