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

The `embedding` container downloads PLM checkpoints from HuggingFace on
first use. `MODEL_CACHE_DIR` on the host is bind-mounted read-only — on
first boot you can temporarily mount it read-write (or `docker compose run
--rm embedding python -c "import transformers; ..."`) to warm the cache.

The Ankh-CL checkpoint is pointed at by `ANKH_CL_CHECKPOINT` — set this to
either a local path under `MODEL_CACHE_DIR` or a HuggingFace repo id
(`DeepFoldProtein/Ankh-Large-Contrastive`). Downloads land in the HF cache
inside the container; re-mounting keeps them warm across restarts.

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
