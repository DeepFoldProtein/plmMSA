# API spec — `POST /v2/align/pairwise`

Reference for integrators. FASTA-shape pairwise alignment over PLM
embeddings — sibling of [`POST /v2/templates/realign`][realign-spec]
(which re-aligns an existing hmmsearch A3M) and a higher-level
surface over [`POST /v2/align`][align-route] (which takes raw
embeddings).

- **Status**: stable on the `/v2/` surface (URL-namespaced; no
  silent-breaking changes — see [Versioning](#versioning)).
- **OpenAPI**: when the api service is run with `api.openapi_public =
  true` in `settings.toml`, the live schema is at `/openapi.json` and
  Swagger UI at `/docs`. This doc is the human-readable form.

[realign-spec]: ./templates-realign-spec.md
[align-route]: ../src/plmmsa/api/routes/v2.py

---

## 1. Endpoint

```
POST {base_url}/v2/align/pairwise
```

| key         | value                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| Method      | `POST`                                                                                                             |
| Path        | `/v2/align/pairwise`                                                                                               |
| Body        | JSON, UTF-8, see [§3](#3-request-body)                                                                             |
| Response    | JSON, UTF-8, see [§4](#4-response-body)                                                                            |
| Auth        | **public** -- no token required, see [§2](#2-authentication)                                                       |
| Idempotency | pure function of the request body (modulo non-deterministic aligner hyperparameters — see [§7](#7-determinism))    |
| Rate limits | per-IP via the api middleware + Cloudflare WAF; no endpoint-specific limits today                                  |

The endpoint is **synchronous**. Response time is dominated by the
PLM embedding cost — typical single-query + a handful of targets
returns in 1–10 s on a single Ada-class GPU. Larger fan-out scales
linearly with `unique_target_seqs`.

### When to use this vs. `/v2/align`

- `/v2/align` takes **raw embeddings** (the per-residue float tensors
  the embedding service produced). Use it if the caller already has
  embeddings cached client-side.
- `/v2/align/pairwise` takes **FASTA-shape** inputs and runs the
  embed step server-side. Use it for "I have a query sequence and N
  target sequences — give me OTalign on them" — the common case.

### When to use this vs. `/v2/templates/realign`

- `/v2/templates/realign` parses an hmmsearch-style A3M (records carry
  `>id/start-end` headers and a gap-aligned residue row), re-aligns
  each template against the query, and re-stamps the headers in
  hmmsearch shape.
- `/v2/align/pairwise` takes plain `(id, sequence)` pairs with no
  pre-existing alignment. Use it when there is no hmmsearch A3M
  upstream — e.g. you have a query plus a list of candidate sequences
  from any source.

---

## 2. Authentication

**This endpoint is public.** No `Authorization` header is required;
requests without credentials succeed. This is the deliberate split
from the rest of `/v2/`: `/v2/embed`, `/v2/align`, and
`/v2/templates/realign` are bearer-gated (admin-minted tokens), but
`/v2/align/pairwise` is the integration surface for ColabFold and
other external clients and must be reachable without credentials.

If the client sends an `Authorization: Bearer <token>` header anyway,
it is ignored on this route -- the response is identical with or
without it.

Abuse protection is enforced at two layers, neither of which the
client has to think about:

- **Cloudflare WAF + per-IP rate limiting** at the edge on the
  `plmmsa.deepfold.org` hostname (zero code cost; configured in the
  CF dashboard).
- **`api` service in-process rate limiter** as defense in depth -- the
  same per-IP limiter that protects the bearer-gated routes. When the
  limit is exceeded the response is `429` with `Retry-After`, not a
  custom plmMSA error envelope; clients should treat `429` as the
  back-off signal.

Other `/v2/` endpoints retain bearer auth; `/v2/align/pairwise` is the
only intentionally-public surface in this group.

---

## 3. Request body

JSON object. Field types follow the JSON spec (`string`, `integer`,
`number`, `boolean`, `object`, `array`).

| field            | type      | required | default        | constraints                                                                                                                                                          |
| ---------------- | --------- | -------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query_id`       | `string`  | no       | `"query"`      | opaque id; echoed back in `stats.query_id`. Not used for matching; useful for log correlation.                                                                       |
| `query_sequence` | `string`  | yes      | —              | residues, normalized server-side via `upper().replace("-", "")` after whitespace strip; result must be `[A-Z]+` and `len ≤ 1022` (configurable via `PairwiseAlignConfig.max_query_length`) |
| `targets`        | `array`   | yes      | —              | one or more objects of shape `{id, sequence}` (see below). `id`s must be unique within the request.                                                                  |
| `model`          | `string`  | no       | `"ankh_large"` | one of `ankh_cl`, `ankh_large`, `esm1b`, `prott5` (whichever the embedding service has loaded)                                                                       |
| `mode`           | `string`  | no       | `"glocal"`     | OTalign DP mode: `local` / `global` / `glocal` / `q2t` / `t2q`. `glocal` is the operator-recommended default (see `templates-realign-spec.md` §3.1 for the rationale) |
| `aligner`        | `string`  | no       | `"otalign"`    | aligner id. Currently `otalign` is the supported choice; `plmalign` works but is exposed primarily via `/v2/align`                                                   |
| `options`        | `object`  | no       | `{}`           | additional aligner tunables — `eps`, `tau`, `n_iter`, `tol`, `gap_open`, `gap_extend`, `fused_sinkhorn`. Forwarded to the align service unchanged                    |
| `emit_a3m`       | `boolean` | no       | `true`         | include a rendered A3M payload in the response (`a3m` field). Set false to save bandwidth when only `columns` is needed                                              |
| `sort_by_score`  | `boolean` | no       | `false`        | `true` → output records emitted in score-descending order. `false` → preserve input order                                                                            |

### `targets[]` entry

| field      | type     | required | constraints                                                                                                                                  |
| ---------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`       | `string` | yes      | 1–256 chars; opaque to the server; used as the row id in the response and (when `emit_a3m=true`) as the `>id` header in the A3M payload     |
| `sequence` | `string` | yes      | residues; normalized server-side the same way as `query_sequence`; result must be `[A-Z]+` and `len ≤ 1022`                                  |

---

## 4. Response body

### 4.1 200 OK — successful run

Content-Type: `application/json`.

```json
{
  "alignments": [
    {
      "target_id": "t1",
      "score": 6.398,
      "columns": [[0, 0], [1, 1], [-1, 2], [-1, 3], [2, 4], [3, 5]],
      "query_start": 0,
      "query_end": 4,
      "target_start": 0,
      "target_end": 6,
      "a3m_row": "ABxxCD"
    }
  ],
  "stats": { ... },
  "a3m": ">query\nABCD\n>t1 score:6.398\nABxxCD\n"
}
```

| field        | type    | notes                                                                                                                                  |
| ------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `alignments` | array   | one entry per **kept** target (targets where the aligner placed zero residues are dropped from this list — see `stats.targets_dropped_no_match`) |
| `stats`      | object  | machine-readable run summary. See [§4.3](#43-stats-object)                                                                             |
| `a3m`        | string  | rendered A3M payload, present only when `emit_a3m=true` (default). Empty string when no targets survived. Format pinned in [§4.2](#42-a3m-payload-format) |

### 4.1.1 `alignments[]` entry

| field          | type           | notes                                                                                                                                                                                                                  |
| -------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target_id`    | string         | echoes the corresponding `targets[].id` from the request                                                                                                                                                               |
| `score`        | number         | aligner's headline score (PMI-normalized DP path score for OTalign)                                                                                                                                                    |
| `columns`      | `[[qi,ti]]`    | the alignment path. Each pair is one column. `qi` is a 0-based query residue index or `-1` for "gap on the query side". `ti` is the analogous target index. Indices are into the **normalized** sequences (post-uppercase, post-gap-strip) |
| `query_start`  | integer        | first matched query index (0-based, **inclusive**)                                                                                                                                                                     |
| `query_end`    | integer        | one past the last matched query index (**exclusive**, half-open with `query_start`)                                                                                                                                    |
| `target_start` | integer        | first matched target index (0-based, inclusive)                                                                                                                                                                        |
| `target_end`   | integer        | one past the last matched target index (exclusive)                                                                                                                                                                     |
| `a3m_row`      | string \| null | rendered A3M row — uppercase residue or `-` per query column, plus lowercase insertions kept between the first and last matched query column. `len(a3m_row) >= len(query_sequence)`. `null` when `emit_a3m=false`     |

### 4.2 A3M payload format

When `emit_a3m=true`, the payload is a ColabFold/AlphaFold-shape A3M:

```
>{query_id}
{query_sequence}
>{target_id} score:{score:.3f}
{a3m_row}
>{target_id_2} score:{score:.3f}
{a3m_row_2}
...
```

Three invariants:

1. **The query is the first record** (`>{query_id}` + the normalized
   query sequence). Downstream tools that consume A3M (ColabFold,
   AlphaFold, OpenFold, Boltz, Protenix) expect the query at index 0
   — the payload is ready to feed in directly. The query record is
   always present, even when every target was dropped as no-match.
2. **Target rows preserve middle insertions, trim ends.** Each row
   has `len(query_sequence)` aligned slots (uppercase residue per
   matched query column, `-` per gap). Lowercase **insertions**
   (`(qi=-1, ti>=0)` columns) are kept **between** the first and last
   matched query column; insertions before the first match or after
   the last match are dropped. Motivation: the aligner often emits
   noisy fragments outside the matched span; downstream MSA
   consumers don't want them in the row, but the central insertions
   *are* informative.
3. **`score:N.NNN`** (lowercase, colon separator, three decimal
   places) is appended to every target header. The query header
   carries no score token.

### 4.3 Stats object

```json
{
  "pipeline": "align_pairwise",
  "query_id": "q",
  "query_length": 649,
  "targets_in": 12,
  "targets_kept": 11,
  "targets_dropped_no_match": 1,
  "unique_target_seqs": 10,
  "model": "ankh_large",
  "mode": "glocal",
  "aligner": "otalign",
  "sort_by_score": false,
  "emit_a3m": true
}
```

`targets_in == targets_kept + targets_dropped_no_match` is an
invariant.

---

## 5. Errors

All non-2xx responses return JSON of shape:

```json
{
  "code": "E_<STABLE_CODE>",
  "message": "human-readable message",
  "detail": { ... }
}
```

`code` is part of the public contract — clients should branch on the
code, never the message.

| HTTP | code              | when                                                                                                                                                                                                                           | typical `detail` keys                  |
| ---- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| 400  | `E_INVALID_FASTA` | empty / non-amino-acid query or target, or duplicate `target_id`s                                                                                                                                                              | `target_id`                            |
| 400  | `E_SEQ_TOO_LONG`  | query OR any target residue count exceeds `max_query_length` (default 1022)                                                                                                                                                    | `length`, `max`, optional `target_id`  |
| 413  | `E_QUEUE_FULL`    | `targets` longer than `PairwiseAlignConfig.max_records` (default 5000)                                                                                                                                                         | `records`, `max`                       |
| 429  | (no envelope)     | per-IP rate limit exceeded (api middleware or Cloudflare WAF). Body is the standard rate-limit response, not the plmMSA error envelope; honor `Retry-After`                                                                    | n/a                                    |
| 422  | (FastAPI default) | request body fails Pydantic validation (missing required field, wrong type, etc.). Returned by FastAPI before the orchestrator runs; body uses FastAPI's standard validation-error format, not the plmMSA error envelope above | n/a                                    |
| 502  | `E_INTERNAL`      | upstream embedding / align service returned an unexpected response shape or was unreachable                                                                                                                                    | `service`, `cause`, `expected`, `got`  |
| 503  | `E_GPU_OOM`       | embedding service OOM'd while encoding                                                                                                                                                                                         | `model`, `cause`                       |

The full set of stable error codes lives in
[`src/plmmsa/errors.py`](../src/plmmsa/errors.py).

---

## 6. Examples

### 6.1 curl — query + 2 targets

No `Authorization` header is needed -- the endpoint is public.

```bash
curl -sS -X POST "$BASE_URL/v2/align/pairwise" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "query_id":       "q",
  "query_sequence": "MKTIIALSYIFCLVFA",
  "targets": [
    { "id": "t1", "sequence": "MKTIIALSYIFCLVF" },
    { "id": "t2", "sequence": "AKTIIALSWVFCLVFA" }
  ],
  "sort_by_score": true
}
JSON
```

Build the body from FASTA files with `jq`:

```bash
jq -n \
  --arg q "$(grep -v '^>' query.fasta | tr -d '\n')" \
  --slurpfile t <(jq -Rs .  < targets.fasta) \
  '...'  # see the Python example below for the cleaner version
```

(For non-trivial inputs the Python example is far more readable —
parsing FASTA in `jq` is painful.)

### 6.2 Python `requests`

```python
import os
import requests


def read_fasta(path: str) -> list[tuple[str, str]]:
    """Minimal FASTA reader: yields `(id, sequence)` pairs."""
    out: list[tuple[str, str]] = []
    rid, buf = None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if rid is not None:
                out.append((rid, "".join(buf)))
            rid = line[1:].split()[0]
            buf = []
        elif line:
            buf.append(line)
    if rid is not None:
        out.append((rid, "".join(buf)))
    return out


(query_id, query_seq), *_ = read_fasta("query.fasta")
targets = [{"id": tid, "sequence": s} for tid, s in read_fasta("targets.fasta")]

resp = requests.post(
    f"{os.environ.get('BASE_URL', 'https://plmmsa.deepfold.org')}/v2/align/pairwise",
    json={
        "query_id":       query_id,
        "query_sequence": query_seq,
        "targets":        targets,
        "sort_by_score":  True,
    },
    timeout=900,
)
resp.raise_for_status()
body = resp.json()

print(body["stats"])
for a in body["alignments"]:
    print(f"{a['target_id']:>20s}  score={a['score']:7.3f}  "
          f"q[{a['query_start']}:{a['query_end']}]  "
          f"t[{a['target_start']}:{a['target_end']}]")
```

A ready-to-run version of this script ships under
[`bin/align_pairwise_client.py`](../bin/align_pairwise_client.py).
It defaults `--base-url` to `https://plmmsa.deepfold.org` (the
production endpoint) and also accepts `--chunk-size N` to transparently
split target lists larger than the server's `max_records` cap into
multiple `/v2/align/pairwise` calls, merging the responses into one A3M
(query header at the top, target rows in input order, or globally
score-sorted when `--sort-by-score` is set). An optional `--out-hist
PATH` lazy-imports matplotlib and writes a PNG histogram of the
returned scores; the rest of the script remains stdlib-only.

### 6.3 Error response

```json
{
  "code": "E_INVALID_FASTA",
  "message": "duplicate target_id 't1'.",
  "detail": { "target_id": "t1" }
}
```

---

## 7. Determinism

OTalign's Sinkhorn solver is deterministic on a given device +
tensor layout. Same inputs → same `alignments` byte-for-byte across
reruns **within the same plmMSA build**. The same two caveats apply
as for `/v2/templates/realign` (cross-GPU float drift; cross-version
hyperparameter changes) — see
[`templates-realign-spec.md` §7](./templates-realign-spec.md#7-determinism).

---

## 8. Versioning

This endpoint is on the **`/v2/` surface**: URL-namespaced, no silent
breaking changes. The `stats` object is append-only; the set of
error codes is additive; the column / span convention is part of the
contract.

---

## 9. Limits + sizing

| limit                                  | default | tunable                                       |
| -------------------------------------- | ------: | --------------------------------------------- |
| max query residues                     |    1022 | `PairwiseAlignConfig.max_query_length`        |
| max target residues per entry          |    1022 | same                                          |
| max targets per request                |    5000 | `PairwiseAlignConfig.max_records`             |
| request timeout (embed + align total)  |   900 s | `PLMMSA_PAIRWISE_TIMEOUT_S` env var           |

Embedding is the bottleneck. Identical target residue strings are
de-duplicated server-side before embedding, so requests that include
the same canonical chain under multiple ids run as fast as the
unique-sequence count suggests. Watch `stats.unique_target_seqs` to
see the dedup payoff.

---

## 10. See also

- [`templates-realign-spec.md`](./templates-realign-spec.md) — sister
  endpoint for re-aligning an hmmsearch A3M.
- [`submitting-msa.md`](./submitting-msa.md) — `/v2/msa` for
  generating an MSA from scratch.
- `src/plmmsa/align/pairwise.py` — orchestrator source.
- `src/plmmsa/errors.py` — full enumeration of stable error codes.
- `/openapi.json` on a running api service — machine-readable schema.
