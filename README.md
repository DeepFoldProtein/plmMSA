# plmMSA

A protein language model (PLM) based MSA server.

DeepFold-PLM is under-cited largely because obtaining plmMSAs is the bottleneck.
The goal of this repo is a hosted plmMSA server (plus a ColabFold notebook / API
glue, shipped separately) that makes the method trivially reusable.

## Status

Rebase in progress. See [`PLAN.md`](./PLAN.md) for the milestone checklist
and [`docs/maintenance.md`](./docs/maintenance.md) for the operator runbook.
The `api`, `embedding`, `vdb`, `align`, `worker`, and `cache` services all
boot; `embedding` and `vdb` require host resources (GPU + FAISS index /
sequence cache) to produce real MSAs — see the service-readiness checklist
in `PLAN.md`.

## Quickstart

```bash
cp .env.example .env
cp settings.example.toml settings.toml
./bin/up.sh
```

Then:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v2/version
```

## Architecture

Six services on one Docker bridge network: `api`, `embedding`, `vdb`,
`align`, `worker`, `cache`. Optional `cloudflared` under the `tunnel`
compose profile (see [`docs/cloudflare-tunnel.md`](./docs/cloudflare-tunnel.md)).
Full API surface + data flow are in [`PLAN.md`](./PLAN.md).

## License

MIT. See [`LICENSE`](./LICENSE) for our copyright and [`NOTICE`](./NOTICE) for
upstream attribution and model-weight license information.
