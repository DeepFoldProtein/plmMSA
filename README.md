# plmMSA

A protein language model (PLM) based MSA server.

DeepFold-PLM is under-cited largely because obtaining plmMSAs is the bottleneck.
The goal of this repo is a hosted plmMSA server (plus a ColabFold notebook / API
glue, shipped separately) that makes the method trivially reusable.

## Quickstart

**Client** (hitting a running server):

- Full recipe with curl + Python examples, errors, and limits:
  [`docs/submitting-msa.md`](./docs/submitting-msa.md).
- TL;DR — `/v2/msa` is anonymous; no token needed for MSA submission.

  ```bash
  # Submit an MSA job. With no `models` field, the server aggregates every
  # enabled PLM that has a VDB collection (today: ankh_cl + esm1b) and
  # unions the hits by target id.
  curl -X POST https://plmmsa.deepfold.org/v2/msa \
    -H "Content-Type: application/json" \
    -d '{"sequences": ["MKTIIAL..."], "k": 500}'

  # Poll (status: queued → running → succeeded) and extract A3M
  curl https://plmmsa.deepfold.org/v2/msa/<job_id> \
    | jq -r '.result.payload' > query.a3m
  ```

  Tokens are only needed for `/v2/embed`, `/v2/search`, `/v2/align`, and
  `/admin/*` — the raw-service passthroughs and admin API.

**Operator** (standing up the stack):

```bash
cp .env.example .env            # edit for your host
cp settings.example.toml settings.toml
./bin/up.sh
curl http://localhost:8080/health
curl http://localhost:8080/v2/version
```

See [`docs/maintenance.md`](./docs/maintenance.md) for the operator runbook
and [`docs/cloudflare-tunnel.md`](./docs/cloudflare-tunnel.md) for publishing
under your own hostname.

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

## Observability

- Structured JSON logs (`plmmsa.access`, `plmmsa.audit`, service loggers)
- `X-Request-ID` stamped on every response and forwarded to sidecars
- Prometheus `/metrics` exposing request count, latency, in-flight gauge

## License

MIT. See [`LICENSE`](./LICENSE) for our copyright and [`NOTICE`](./NOTICE) for
upstream attribution and model-weight license information.
