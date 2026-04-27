# Submitting an MSA job

This is the end-to-end recipe for getting an A3M out of a running plmMSA
server. If you operate the server yourself, pair this doc with
[`docs/maintenance.md`](./maintenance.md) (setup) and
[`docs/cloudflare-tunnel.md`](./cloudflare-tunnel.md) (public edge).

**Endpoint hosts used in this doc**

- Public (DeepFold-hosted): `https://plmmsa.deepfold.org`
- Local operator loop: `http://localhost:8080`

**Auth model**

- `/v2/msa` (submit / poll / cancel), `/v2/version`, `/health`, `/metrics`
  — **anonymous**. No token required. Abuse is bounded by the per-IP rate
  limiter (`settings.ratelimit.per_ip_rpm`) and queue backpressure.
- `/v2/embed`, `/v2/search`, `/v2/align` — **require a Bearer token**
  (these are internal service passthroughs; they skip the queue / rate
  limits that protect `/v2/msa` and so have their own gate).
- `/admin/*` — **require a Bearer token** and must never be published.

If you just want MSAs, you can skip step 1. Only come back for a token if
you need the pass-through `/v2/embed`, `/v2/search`, or `/v2/align`
endpoints.

---

## 1. (Optional) Get a client token

You only need this for `/v2/embed`, `/v2/search`, `/v2/align`, or
`/admin/*`. Only the operator (you, if you're running the stack) can mint
tokens. From the host running Docker Compose:

```bash
# Mint a token labeled "my-lab". Shown ONCE — save it now.
curl -sS -X POST http://localhost:8080/admin/tokens \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"label":"my-lab","rate_limit_rpm":60}' | jq
```

Response:

```json
{
  "token": "bAsmB6IXvi8Yc...",
  "record": {
    "id": "abc-123",
    "label": "my-lab",
    "created_at": 1713782400.0,
    "expires_at": 1721558400.0,
    "revoked": false,
    "rate_limit_rpm": 60
  }
}
```

**Notes**

- `ADMIN_TOKEN` is the bootstrap token from your `.env`. Keep it for
  administration; hand out minted tokens to callers instead.
- Tokens get a default 90-day expiry (controlled by
  `settings.api.default_token_ttl_s`). Pass `"expires_at": 0` to mint a
  non-expiring token; pass `"expires_at": <unix-ts>` to pick your own.
- `rate_limit_rpm` overrides the default per-token cap
  (`settings.ratelimit.per_token_rpm`, 120/min out of the box).
- List tokens: `GET /admin/tokens`. Revoke: `DELETE /admin/tokens/{id}`.
- `/admin/*` must stay internal. The public Cloudflare tunnel hostname
  should return 404 for that prefix — see `docs/cloudflare-tunnel.md`.

---

## 2. Submit an MSA job

Jobs are async: `POST /v2/msa` returns a `job_id` immediately; the worker
runs the embed → VDB search → fetch → re-embed → align → A3M pipeline.
No token needed.

**Default behavior: aggregate across every PLM with a VDB collection.**
Today that's `ankh_cl` + `esm1b` — the server runs both in parallel, then
unions the hits by target id (keeping the best-scoring alignment per
sequence). You only need to set `models` if you want to narrow or reorder
the list.

```bash
# Simplest submission: server picks the PLMs.
curl -sS -X POST https://plmmsa.deepfold.org/v2/msa \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["MKTIIALSYIFCLVFADYKDDDDK"],
    "output_format": "a3m",
    "k": 500
  }' | jq
```

Explicit single PLM (the old escape hatch):

```bash
curl -sS -X POST https://plmmsa.deepfold.org/v2/msa \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKT..."], "models": ["ankh_cl"], "k": 500}'
```

Pick a subset for comparison:

```bash
curl -sS -X POST https://plmmsa.deepfold.org/v2/msa \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKT..."], "models": ["ankh_cl", "esm1b"], "k": 500}'
```

Response (`202 Accepted`):

```json
{
  "job_id": "a7d8...",
  "status": "queued",
  "status_url": "/v2/msa/a7d8..."
}
```

### Submit fields (`/v2/msa`)

| field             | required | default             | notes                                                                                                                                                                                                                                                     |
| ----------------- | -------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sequences`       | yes      | —                   | One string per chain. 1–N chains. AA alphabet only.                                                                                                                                                                                                       |
| `models`          | no       | enabled PLMs w/ VDB | List of PLM ids run in parallel; hits unioned by target id. Today: `ankh_cl, esm1b`.                                                                                                                                                                      |
| `model`           | no       | —                   | Legacy single-PLM shortcut. Equivalent to `models: ["<value>"]`.                                                                                                                                                                                          |
| `output_format`   | no       | `a3m`               | `a3m`. Stockholm / FASTA export is a deferred item.                                                                                                                                                                                                       |
| `paired`          | no       | `false`             | Paired-by-TaxID MSA across chains. Requires ≤16 chains.                                                                                                                                                                                                   |
| `query_id`        | no       | `query`             | FASTA header id in the emitted A3M.                                                                                                                                                                                                                       |
| `collection`      | no       | `<model>_uniref50`  | Single-PLM collection override. Ignored when `models` has >1 entry.                                                                                                                                                                                       |
| `collections`     | no       | `{}`                | Per-model VDB collection map: `{"ankh_cl":"ankh_uniref50", ...}`.                                                                                                                                                                                         |
| `k`               | no       | `1000`              | FAISS neighbors to fetch **per model**. Typical 500–2000; max 10000.                                                                                                                                                                                      |
| `aligner`         | no       | `plmalign`          | `plmalign`, `plm_blast`, or `otalign`.                                                                                                                                                                                                                    |
| `mode`            | no       | `local`             | `local` / `global` (all aligners) · `glocal` / `q2t` / `t2q` (OTalign only; other aligners 400 on these). No silent remap — OTalign honors the mode verbatim.                                                                                             |
| `score_model`     | no       | aligner default     | PLM used to build the score matrix. `plmalign`/`plm_blast` default to `prott5` (shard store); `otalign` defaults to `ankh_large` (live re-embed). Pass `""` to disable cross-PLM scoring.                                                                 |
| `filter_by_score` | no       | `true`              | Apply the post-alignment score-threshold filter. Cutoff is aligner-specific: PLMAlign / pLM-BLAST use upstream Algorithm 1 step 5 (`min(0.2·len(Q), 8.0)`); OTalign uses its calibrated transport-mass floor (`[aligners.otalign].filter_threshold`, default `0.25`). Per-aligner enable is in `[aligners.*].filter_enabled`. Per-request `false` always wins. |
| `options`         | no       | `{}`                | Aligner kwargs: `gap_open`, `gap_extend`, `normalize`, ...                                                                                                                                                                                                |
| `force_recompute` | no       | `false`             | Bypass the completed-MSA result cache and always run the pipeline. On success the fresh result still overwrites the cache entry. Useful for reproducing or retrying after a pipeline change.                                                              |

**How multi-model aggregation works.** Each model runs its own pipeline
(embed → search → fetch → embed targets → align) in parallel. After all
finish, the server unions the per-model `AlignmentHit`s by `target_id`
and keeps the highest-scoring variant. The emitted A3M is a single
alignment in the query coordinate frame; `result.stats.per_model` shows
the per-model breakdown (`hits_found`, `hits_fetched`, optional `error`).
A single model's failure does not fail the whole request — the union
returns whatever the surviving models produced.

### Response stats

`result.stats` includes the filter trace:

| field              | meaning                                                                                                                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hits_pre_filter`  | Aligned hit count before the post-alignment score filter.                                                                                                                                                                     |
| `hits_post_filter` | Count after the filter (same as `hits_fetched`).                                                                                                                                                                              |
| `filter_by_score`  | Request-level flag (`filter_by_score` field on the submit).                                                                                                                                                                   |
| `filter_applied`   | `true` only when both the request flag and the aligner's `filter_enabled` are true.                                                                                                                                           |
| `filter_threshold` | Aligner-specific cutoff actually used: `min(0.2 · len(Q), 8.0)` for PLMAlign / pLM-BLAST (dot-product scale), the configured fixed floor for OTalign (`0.25` by default, transport-mass scale).                               |
| `cache_hit`        | `true` when the server served this MSA from the result cache (`cache-emb`) instead of running the pipeline. Absent on fresh compute. The job record's `started_at` and `finished_at` collapse to the same timestamp on a hit. |

A3M rows carry the raw alignment score in the FASTA header
(`>target_id   123.456`), so you can sort or re-filter client-side.

### Limits enforced at the edge

| limit                            | default                  | error code                              |
| -------------------------------- | ------------------------ | --------------------------------------- |
| `limits.max_residues_per_chain`  | 1022                     | `E_SEQ_TOO_LONG` (400)                  |
| `limits.max_chains_paired`       | 16                       | `E_TOO_MANY_CHAINS` (400)               |
| `limits.max_body_bytes`          | 10 MB                    | `E_INVALID_FASTA` (413)                 |
| non-AA characters in `sequences` | —                        | `E_INVALID_FASTA` (400)                 |
| unknown `model`                  | —                        | `E_UNSUPPORTED_MODEL` (400)             |
| `ratelimit.per_ip_rpm`           | 60                       | `E_RATE_LIMITED` (429)                  |
| `ratelimit.per_token_rpm`        | 120 (override per token) | `E_RATE_LIMITED` (429)                  |
| `queue.backpressure_threshold`   | 50                       | `E_QUEUE_FULL` (503, `Retry-After: 5`)  |
| `queue.max_queue_depth`          | 200                      | `E_QUEUE_FULL` (503, `Retry-After: 30`) |

Tune the defaults in `settings.toml` — each section has a one-line
comment describing the knob.

---

## 3. Poll for results

```bash
JOB=a7d8...

# Wait until status is `succeeded` (or `failed` / `cancelled`).
curl -sS https://plmmsa.deepfold.org/v2/msa/$JOB | jq
```

Lifecycle returned by `GET /v2/msa/{job_id}`:

- `queued` — accepted, waiting for the worker
- `running` — worker claimed it; the embed → VDB → align pipeline is live
- `succeeded` — `result.payload` contains the A3M string
- `failed`    — `error.{code,message}` explains why
- `cancelled` — user called `DELETE /v2/msa/{job_id}` before it finished

Typical turn-around on the reference host (single A100/RTX 6000 Ada) for a
~150-residue single-chain query at `k=500`: **30–90 s**. Cold PLM loads on
the first call of the day add another ~20 s.

### Extract the A3M

```bash
curl -sS https://plmmsa.deepfold.org/v2/msa/$JOB \
  | jq -r '.result.payload' > query.a3m
```

The file is a3m-formatted: query is the first record, each hit has a
`>target_id score=...` header followed by the aligned residues in the
query's coordinate frame.

### Cancel an in-flight job

```bash
curl -sS -X DELETE https://plmmsa.deepfold.org/v2/msa/$JOB
# 204 No Content
```

Cancellation is cooperative — the worker checks the cancelled flag between
stages and will still emit stats on whatever completed. Cancel is anonymous
like submit; the practical guard against drive-by cancellations is that
job ids are UUID4s, i.e. not enumerable.

---

## 4. Idempotency, retries, and request IDs

### Idempotency

Two identical `POST /v2/msa` calls from the same caller within 10 minutes
return the same `job_id`. The scope is `token_id` when authenticated,
otherwise the client IP. The key is `sha256(scope, canonical_payload)`.
This means a client retry after a flaky network — even one that saw a
partial response — won't double-submit.

Changing any field (model, k, options, sequence) produces a new job.
NAT-fronted clients may de-dup across members sharing an egress IP; they
would have received identical answers anyway, so this is harmless.

### Retries

- `429 E_RATE_LIMITED` — honor `Retry-After` (seconds). Token-scoped and
  IP-scoped use the same code; the `detail.scope` field tells you which.
- `503 E_QUEUE_FULL` — the queue is under backpressure; honor `Retry-After`.
- `5xx` other — exponential backoff; safe to retry (idempotent above).

### Request IDs

Every response carries `X-Request-ID`. Send your own value if you want it
threaded through the access log and the sidecar logs:

```bash
curl -sS https://plmmsa.deepfold.org/v2/msa \
  -H "X-Request-ID: my-client-trace-001" \
  -H "Content-Type: application/json" \
  -d @request.json
```

When you file a bug, include the `X-Request-ID` — operators can grep
every log line the request touched.

---

## 5. Error taxonomy

All errors return this shape:

```json
{
  "code": "E_SEQ_TOO_LONG",
  "message": "sequences[0] is 2048 residues; max is 1022.",
  "detail": { "chain": 0, "length": 2048, "max": 1022 }
}
```

Stable codes (machine-readable, versioned):

| code                  | http    | meaning                                               |
| --------------------- | ------- | ----------------------------------------------------- |
| `E_SEQ_TOO_LONG`      | 400     | chain longer than `max_residues_per_chain`            |
| `E_TOO_MANY_CHAINS`   | 400     | paired MSA exceeded `max_chains_paired`               |
| `E_INVALID_FASTA`     | 400/413 | non-AA chars, empty sequence, or body too big         |
| `E_UNSUPPORTED_MODEL` | 400     | `model` not enabled in `settings.toml`                |
| `E_AUTH_MISSING`      | 401     | no Bearer header                                      |
| `E_AUTH_INVALID`      | 401     | token not found / revoked / expired                   |
| `E_RATE_LIMITED`      | 429     | per-IP or per-token RPM exceeded                      |
| `E_QUEUE_FULL`        | 503     | queue at `backpressure_threshold` / `max_queue_depth` |
| `E_JOB_NOT_FOUND`     | 404     | `/v2/msa/{id}` unknown id                             |
| `E_GPU_OOM`           | 503     | embedding service ran out of GPU                      |
| `E_INTERNAL`          | 5xx     | upstream sidecar unreachable / non-JSON               |
| `E_GONE`              | 410     | `/v1/*` was sunset                                    |

---

## 6. Python example

```python
import time

import requests

BASE = "https://plmmsa.deepfold.org"


def submit(
    sequence: str,
    *,
    models: list[str] | None = None,  # None = server default (aggregate)
    k: int = 500,
) -> str:
    payload: dict = {"sequences": [sequence], "k": k}
    if models is not None:
        payload["models"] = models
    r = requests.post(
        f"{BASE}/v2/msa",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["job_id"]


def wait(job_id: str, *, poll_s: float = 3.0, timeout_s: float = 600) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = requests.get(f"{BASE}/v2/msa/{job_id}", timeout=10)
        r.raise_for_status()
        job = r.json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(poll_s)
    raise TimeoutError(f"Job {job_id} did not finish in {timeout_s}s")


if __name__ == "__main__":
    jid = submit("MKTIIALSYIFCLVFADYKDDDDK")
    job = wait(jid)
    if job["status"] != "succeeded":
        raise SystemExit(f"job failed: {job.get('error')}")
    with open("query.a3m", "w") as fh:
        fh.write(job["result"]["payload"])
```

---

## 7. Health, version, and metrics

```bash
curl https://plmmsa.deepfold.org/v2/version        # model + API versions (anonymous)
curl https://plmmsa.deepfold.org/health            # liveness
curl http://localhost:8080/metrics                 # Prometheus text exposition (operator-only)
```

`/v2/version` is anonymous on purpose: clients use it to discover which
PLMs / collections are enabled before picking a `model`.

---

## 8. Recipes

### Fast / low-latency (single PLM)

```json
{"sequences": ["MKT..."], "models": ["ankh_cl"], "k": 100}
```

### Default aggregate (what you get if you omit `models`)

```json
{"sequences": ["MKT..."], "k": 500}
```

### Deep / high-coverage aggregate

```json
{"sequences": ["MKT..."], "models": ["ankh_cl", "esm1b"], "k": 1000, "mode": "local"}
```

### Single PLM for sensitivity comparison

```json
{"sequences": ["MKT..."], "models": ["esm1b"], "k": 500}
```

### OTalign with explicit DP mode (glocal / q2t / t2q)

OTalign honors all five DP modes verbatim. `mode` is passed straight
through to the DP — no silent substitution.

```json
{"sequences": ["MKT..."], "aligner": "otalign", "mode": "glocal"}
```

| mode     | when to pick                                                      |
| -------- | ----------------------------------------------------------------- |
| `local`  | SW-style. Free start + end, negative-floor clamp.                 |
| `global` | NW. Pay end-gap cost on both sides.                               |
| `glocal` | Semi-global. Free end-gaps both sides (upstream OTalign default). |
| `q2t`    | Free query end-gap only (template is global).                     |
| `t2q`    | Free template end-gap only (query is global).                     |

PLMAlign / pLM-BLAST only support `local` / `global` and 400 on the three
OTalign-only modes.

### Custom gap penalties

```json
{"sequences": ["MKT..."], "options": {"gap_open": 10.0, "gap_extend": 1.0}}
```

### Override the score-matrix mode per request

PLMAlign can score the query-vs-target residue-pair matrix three ways.
Default follows upstream PLMAlign. Override per job via `options.score_matrix`:

| mode                   | what it does                                                                                                  | typical use                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `dot_zscore` (default) | raw `q @ t.T` then global Z-score `(S - mean)/(std + 1e-3)` — centers the matrix around 0 with ~unit variance | same convention as upstream PLMAlign; scores are comparable across jobs |
| `cosine`               | L2-normalize each per-residue vector, then dot product. Scores live in `[-1, 1]`                              | easier to reason about for gap tuning; robust to PLM-magnitude drift    |
| `dot`                  | raw dot product, no normalization                                                                             | debug / research only — score scale tracks embedding magnitudes         |

Example:

```json
{
  "sequences": ["MKT..."],
  "options": {"score_matrix": "cosine", "gap_open": 10.0, "gap_extend": 1.0}
}
```

All three modes run our affine-gap SW/NW traceback. The upstream pLM-BLAST
multi-path SW is a **different algorithm** (not a scoring variant) and is
tracked as deferred work in `PLAN.md`; when it lands it will be selectable
via a different aligner id, not via `score_matrix`.

### Cross-PLM scoring (planned, not yet wired)

`aligners.plmalign.score_model` in `settings.toml` is the future knob for
"search with PLM A, align with PLM B" — handy for comparing search
sensitivity of one model against the alignment quality of another. Costs
one extra embedding pass per hit. Today it's a no-op; see PLAN.md.


### Per-model collection override

```json
{
  "sequences": ["MKT..."],
  "models": ["ankh_cl", "esm1b"],
  "collections": {
    "ankh_cl": "ankh_uniref50",
    "esm1b": "esm1b_uniref50"
  },
  "k": 300
}
```

---

## 9. Debugging checklist

1. `GET /v2/version` — does the server know the model you asked for?
2. `GET /health` — is the API up and can it reach cache-ops?
3. The job is stuck in `queued` for a long time → operator-side: is the
   worker container running? `docker compose logs worker --tail 200`.
4. `failed` with `E_INTERNAL detail.service=embedding` → GPU service is
   sick. `docker compose logs embedding --tail 200`.
5. Repeated `E_RATE_LIMITED` → check your token's `rate_limit_rpm`. Ask the
   operator to mint a new token with a higher cap.
6. 413 with `E_INVALID_FASTA` on a legitimate multi-chain request →
   `limits.max_body_bytes` is too small. Raise it in `settings.toml`.

Attach the `X-Request-ID` of the failing call when filing a bug.
