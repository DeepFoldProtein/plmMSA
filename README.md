# plmMSA

A protein language model (PLM) based MSA server. Drop-in replacement for
MMseqs2 MSA generation in ColabFold / Boltz / Protenix, powered by
Ankh-CL + ESM-1b retrieval and ProtT5 / Ankh-Large pairwise alignment.

DeepFold-PLM is under-cited largely because obtaining plmMSAs is the
bottleneck — a hosted server makes the method trivially reusable.

---

## For end users

Public endpoint: **`https://plmmsa.deepfold.org`** (anonymous; no token
required for MSA submission).

### Submit a job in the browser

The static web UI lives at
[`https://plmmsa.deepfold.org/ui`](https://plmmsa.deepfold.org/ui) —
paste a sequence, pick an aligner / mode, click Submit. Job ids are
cached in `localStorage`; copy the URL to share (`?job=<uuid>` loads
that specific job in a fresh browser). No accounts, no build step on
our side either — it's plain HTML / JS served by FastAPI.

### One-shot curl

```bash
# Submit an MSA job.
JOB=$(curl -sS -X POST https://plmmsa.deepfold.org/v2/msa \
    -H "Content-Type: application/json" \
    -d '{"sequences": ["MKTIIAL..."]}' | jq -r .job_id)

# Poll until status is "succeeded"; result.payload holds the A3M.
while [[ "$(curl -sS https://plmmsa.deepfold.org/v2/msa/$JOB | jq -r .status)" \
          != "succeeded" ]]; do sleep 5; done

curl -sS https://plmmsa.deepfold.org/v2/msa/$JOB \
    | jq -r .result.payload > query.a3m
```

Every knob (aligner, mode, filter, paired, score model, VDB collection,
output format) is optional — the server picks sensible defaults. Full
recipe with the request-field table, error codes, and Python examples:
[`docs/submitting-msa.md`](./docs/submitting-msa.md).

### Drop-in MSA server for folding engines

plmMSA speaks the **ColabFold MsaServer wire shape** under
`/v2/colabfold/{plmmsa,otalign}/*` — point any engine's MSA-server URL
at one of those prefixes and it Just Works. Two flavors:

- `/v2/colabfold/plmmsa` — PLMAlign (fast, ProtT5 shard-backed).
- `/v2/colabfold/otalign` — OTalign (Sinkhorn + position-specific DP).

| Engine                                                                       | Recipe                                                                                                |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **ColabFold** ([`colabfold_batch`](https://github.com/sokrypton/ColabFold))  | `colabfold_batch --host-url https://plmmsa.deepfold.org/v2/colabfold/plmmsa input.fasta out/`         |
| **Boltz** ([`boltz predict`](https://github.com/jwohlwend/boltz))            | `boltz predict --use_msa_server --msa_server_url https://plmmsa.deepfold.org/v2/colabfold/plmmsa ...` |
| **Protenix** ([`bytedance/Protenix`](https://github.com/bytedance/Protenix)) | Set the MSA-server URL in the inference config to `https://plmmsa.deepfold.org/v2/colabfold/plmmsa`.  |

Swap `.../plmmsa` for `.../otalign` to use OTalign instead. Paired MSA
(multimer) support lands with the paired-MSA feature (see
[`PLAN.md`](./PLAN.md)); single-chain MSAs work now.

### Re-align an existing hmmsearch A3M

If you already have a candidate template set (e.g. an HMM scan against
PDB) and want PLM-driven column placements rather than HMMER's,
`POST /v2/templates/realign` runs OTalign / Ankh-Large / glocal over
each template. Output is an A3M whose rows are exactly `query_len`
chars from `[A-Z-]` (no lowercase insertions), with `Score=` stamped
adjacent to a re-intervalled `/start-end`. Bearer-gated. Recipe:
[`docs/templates-realign.md`](./docs/templates-realign.md).

### Response format

plmMSA returns A3M by default — the same format ColabFold / AlphaFold
consume directly. The server stats block (`result.stats`) includes
`hits_found`, `hits_pre_filter`, `hits_post_filter`, `filter_applied`,
and `filter_threshold` so you can see what the pipeline kept vs. what
got dropped by the Algorithm 1 step 5 filter.

---

## For operators

```bash
cp .env.example .env                      # edit for your host
cp settings.example.toml settings.toml
./bin/up.sh
curl http://localhost:8080/healthz         # bare liveness
curl http://localhost:8080/health          # aggregated readiness (all downstreams)
curl http://localhost:8080/v2/version
```

See [`docs/maintenance.md`](./docs/maintenance.md) for the operator
runbook (service knobs, auth tokens, wipe/reload, VDB construction) and
[`docs/cloudflare-tunnel.md`](./docs/cloudflare-tunnel.md) for publishing
the stack under your own hostname.

---

## Architecture

Nine services on one Docker bridge network: `api`, `embedding`, `vdb`,
`align`, `worker`, `cache-ops`, `cache-seq`, `cache-emb`; plus an optional
`cloudflared` under the `tunnel` compose profile. Full API surface + data
flow are in [`PLAN.md`](./PLAN.md).

The public surface (`/v2/*`) enforces input validation (1022 res/chain,
16 chains paired), per-IP + per-token rate limiting, body-size caps,
idempotent submission, queue backpressure, and an audit log. See
[`docs/submitting-msa.md`](./docs/submitting-msa.md) §2 for the full
limits + error-code table.

**Performance** (k=1000 CASP15 targets, current trunk):

| Target | Length | PLMAlign | OTalign |
| ------ | -----: | -------: | ------: |
| T1132  |    102 |    ~10 s |   ~22 s |
| T1104  |    106 |    ~27 s |   ~28 s |
| T1120  |    235 |    ~30 s |   ~42 s |

PLMAlign is roughly 2× faster because ProtT5's shard store + Redis path
index make the target-embedding phase disk-only. OTalign's Ankh-Large
score model still re-embeds per job; a fused-kernel Sinkhorn (`FlashSinkhorn`)
and Ankh-Large shard store are on the roadmap.

## Observability

- Structured JSON logs across every service (`plmmsa.access`,
  `plmmsa.access.{embedding,vdb,align}`, `plmmsa.audit`, per-service
  error loggers). 5xx errors carry the full traceback; 4xx log as
  warnings with method / path / code.
- `X-Request-ID` end-to-end: stamped by api on entry, echoed on every
  response, threaded to embedding / vdb / align on api's internal
  calls, and rebound on the worker when a job is claimed so the
  orchestrator's downstream calls carry the same id. Grep one trace
  with `docker compose logs ... | jq 'select(.request_id == "<rid>")'`.
- Aggregated `/health` fans out to each downstream (HTTP + Redis
  ping) and reports per-service readiness; `/healthz` stays a bare
  liveness probe for compose / CF edge.
- Prometheus `/metrics` on api (request count, latency, in-flight
  gauge). Per-service metrics on embedding/vdb/align/worker is a
  deferred follow-up (see [`PLAN.md`](./PLAN.md)).
- Completed MSAs are cached on `cache-emb` keyed by a canonical
  submit hash — repeat submissions return immediately with
  `result.stats.cache_hit = true`. Clients can opt out with
  `{"force_recompute": true}` on `POST /v2/msa`.

## License

MIT. See [`LICENSE`](./LICENSE) for our copyright and [`NOTICE`](./NOTICE) for
upstream attribution and model-weight license information.
