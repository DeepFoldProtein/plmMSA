# API spec — `POST /v2/templates/realign`

Reference for integrators. Companion to the operator walkthrough at
[`templates-realign.md`](./templates-realign.md), which covers
deployment + rationale; this file is the wire contract.

- **Status**: stable on the `/v2/` surface (URL-namespaced; no
  silent-breaking changes — see [Versioning](#versioning)).
- **OpenAPI**: when the api service is run with `api.openapi_public =
  true` in `settings.toml`, the live schema is at `/openapi.json` and
  Swagger UI at `/docs`. This doc is the human-readable form.

---

## 1. Endpoint

```
POST {base_url}/v2/templates/realign
```

| key         | value                                                                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Method      | `POST`                                                                                                                                                                 |
| Path        | `/v2/templates/realign`                                                                                                                                                |
| Body        | JSON, UTF-8, see [§3](#3-request-body)                                                                                                                                 |
| Response    | JSON, UTF-8, see [§4](#4-response-body)                                                                                                                                |
| Auth        | bearer token, see [§2](#2-authentication)                                                                                                                              |
| Idempotency | every successful run is a pure function of the request body — same body in, same body out (modulo non-deterministic OTalign hyperparameters; see [§7](#7-determinism)) |
| Rate limits | covered by the api service's per-IP / per-token limiter; no endpoint-specific limits today                                                                             |

The endpoint is **synchronous**. Response time is dominated by the PLM
embedding cost — typical single-query, ~600-record inputs return in
10–60 s on a single Ada-class GPU. Future versions may add a
submit-then-poll variant; the sync path is committed to stay.

---

## 2. Authentication

```
Authorization: Bearer <token>
```

Every request must carry a bearer token. Two ways to obtain one:

- The operator-only `ADMIN_TOKEN` env var on the api service (used by
  the operator themselves).
- A scoped client token minted via the admin API (`POST
  /admin/tokens`); see [`submitting-msa.md`](./submitting-msa.md) for
  the recipe.

A missing or unknown token returns `401` with code `E_AUTH_MISSING` /
`E_AUTH_INVALID` (see [§5](#5-errors)). Tokens never appear in
response bodies or logs.

---

## 3. Request body

JSON object. Field types follow the JSON spec (`string`, `integer`,
`number`, `boolean`, `object`).

| field            | type      | required | default        | constraints                                                                                                                                                                                   |
| ---------------- | --------- | -------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query_sequence` | `string`  | yes      | —              | residues, normalized server-side via `upper().replace("-", "")` after whitespace strip; result must be `[A-Z]+` and `len ≤ 1022` (configurable via `TemplatesRealignConfig.max_query_length`) |
| `a3m`            | `string`  | yes      | —              | hmmsearch-style A3M body. ≥ 1 record; every record's `upper+gap` must equal `len(query_sequence)` (post-normalization)                                                                        |
| `query_id`       | `string`  | no       | `"query"`      | reserved for audit/log correlation only. Does **not** appear in the output A3M (the output mirrors hmmsearch's shape — no synthetic query record)                                             |
| `model`          | `string`  | no       | `"ankh_large"` | one of `ankh_cl`, `ankh_large`, `esm1b`, `prott5` (whichever the embedding service has loaded)                                                                                                |
| `mode`           | `string`  | no       | `"glocal"`     | OTalign DP mode: `local` / `global` / `glocal` / `q2t` / `t2q`. **Operators recommend `glocal`** (default) — see [§3.1](#31-mode-default-rationale)                                           |
| `options`        | `object`  | no       | `{}`           | additional OTalign tunables — `eps`, `tau`, `n_iter`, `tol`, `gap_open`, `gap_extend`, `fused_sinkhorn`. Forwarded to the align service unchanged                                             |
| `sort_by_score`  | `boolean` | no       | `false`        | `true` → output records emitted in OTalign-score-descending order (best hit first). `false` → preserve hmmsearch input order (useful for diffing)                                             |

### 3.1 Mode default rationale

A 20-record sample of the bundled CASP-style fixture (1903 template
residues, glocal vs. q2t vs. local):

| mode   | template coverage | hmmsearch agreement | matched-column identity |
| ------ | ----------------: | ------------------: | ----------------------: |
| glocal |             0.949 |               0.595 |                   0.152 |
| q2t    |             0.854 |               0.349 |                   0.127 |
| local  |             0.938 |               0.595 |                   0.155 |

q2t loses ~25 pp on hmmsearch agreement to its forced
`query_end == Lq` artifact (the trailing template residue is pulled
to the last query column). `glocal` is the default; q2t / local /
global / t2q stay callable for callers who need a different end-gap
policy.

---

## 4. Response body

### 4.1 200 OK — successful run

Content-Type: `application/json`. Body is a JSON object:

```json
{
  "format": "a3m",
  "payload": "<output A3M as a single string>",
  "stats": { ... }
}
```

| field     | type     | notes                                                                                        |
| --------- | -------- | -------------------------------------------------------------------------------------------- |
| `format`  | `string` | always `"a3m"` for this endpoint                                                             |
| `payload` | `string` | the re-rendered A3M text (UTF-8, ASCII-safe). Format pinned in [§4.2](#42-output-a3m-format) |
| `stats`   | `object` | machine-readable run summary. Schema in [§4.3](#43-stats-object)                             |

### 4.2 Output A3M format

Three invariants integrators can rely on:

1. **No query record at the top.** The output mirrors hmmsearch's
   shape — only the re-aligned template records. Callers that need
   a ColabFold/AlphaFold-style A3M (query at index 0) prepend it
   client-side: `f">{query_id}\n{query_sequence}\n"` ahead of the
   payload.

2. **Every row is exactly `len(query_sequence)` characters from
   `[A-Z-]`.** No lowercase A3M insertions appear in the output —
   template residues OTalign couldn't place at a query column are
   dropped. Records where OTalign placed zero residues are dropped
   from the output entirely (counted in
   `stats.records_dropped_no_match`).

3. **Headers preserve the input verbatim except for two surgical
   edits.** Worked example:

   ```
   in  : >7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1
   out : >7sch_A/55-703 [subseq from] mol:protein length:720 score:6.398  Exostosin-1
   ```

   - `/start-end` is replaced with `/new_start-new_end` reflecting
     the residues OTalign actually placed (`new_start = orig_start +
     min_kept_ti`, `new_end = orig_start + max_kept_ti`, both 1-based
     inclusive).
   - `score:N.NNN` (lowercase, colon separator, three decimal places)
     is inserted at the end of the technical-tokens section. The
     hmmsearch convention uses a **double-space** to separate
     technical tokens (`mol:protein`, `length:720`) from the
     free-text description (`Exostosin-1`); we slot `score:` right
     before that double-space so it sits next to the other technical
     tokens. Headers without a double-space separator just get
     `score:` appended at the end.
   - Any prior `Score=` / `score:` / `Score:` tokens anywhere in the
     header are stripped before the new one is inserted. Multiple
     prior tokens (rare) are all removed.
   - The domain id (`7sch_A`), `[subseq from]`, `mol:protein`, the
     `length:N` token, and the human-readable description survive
     byte-for-byte.

### 4.3 Stats object

```json
{
  "pipeline": "templates_realign",
  "query_length": 649,
  "records_in": 593,
  "records_kept": 593,
  "records_dropped_sanity": 0,
  "records_dropped_no_match": 0,
  "unique_template_seqs": 93,
  "model": "ankh_large",
  "mode": "glocal",
  "aligner": "otalign",
  "sort_by_score": false
}
```

| field                      | type    | notes                                                                                                      |
| -------------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `pipeline`                 | string  | always `"templates_realign"`                                                                               |
| `query_length`             | integer | length of the normalized query sequence                                                                    |
| `records_in`               | integer | number of records the parser saw, including those dropped by sanity rules                                  |
| `records_kept`             | integer | number of records present in `payload`                                                                     |
| `records_dropped_sanity`   | integer | dropped by the parser (`malformed_header` / `alphabet_error` / `interval_mismatch` / `query_len_mismatch`) |
| `records_dropped_no_match` | integer | OTalign placed zero template residues for these records — they don't appear in `payload`                   |
| `unique_template_seqs`     | integer | distinct template residue strings sent to the embedding service (de-duplicates repeated PDB chains)        |
| `model`                    | string  | the PLM backend that produced the embeddings                                                               |
| `mode`                     | string  | the OTalign DP mode used                                                                                   |
| `aligner`                  | string  | always `"otalign"` for this endpoint today                                                                 |
| `sort_by_score`            | boolean | echoes the request field, so consumers know which ordering they got                                        |

`records_in == records_kept + records_dropped_sanity +
records_dropped_no_match` is an invariant.

---

## 5. Errors

All non-2xx responses return JSON of shape:

```json
{
  "code": "E_<STABLE_CODE>",
  "message": "human-readable message",
  "detail": { ... }   // optional, varies by code
}
```

`code` is part of the public contract — clients should branch on the
code, never the message. The `detail` object is informational; its
keys may change between versions.

| HTTP | code              | when                                                                                                                                                                                                                           | `detail` keys                                       |
| ---- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| 400  | `E_INVALID_FASTA` | empty / non-amino-acid query, or `len(query_sequence)` ≠ a3m's match-state count, or no records survive sanity checks                                                                                                          | `query_length`, `first_drop_reason`, `drop_reasons` |
| 400  | `E_SEQ_TOO_LONG`  | query OR any template residue count exceeds `max_query_length` (default 1022)                                                                                                                                                  | `length`, `max`, optional `target_id`               |
| 401  | `E_AUTH_MISSING`  | `Authorization` header absent                                                                                                                                                                                                  | —                                                   |
| 401  | `E_AUTH_INVALID`  | bearer token unknown, expired, or revoked                                                                                                                                                                                      | —                                                   |
| 413  | `E_QUEUE_FULL`    | a3m has more records than `TemplatesRealignConfig.max_records` (default 5000)                                                                                                                                                  | `records`, `max`                                    |
| 422  | (FastAPI default) | request body fails Pydantic validation (missing required field, wrong type, etc.). Returned by FastAPI before the orchestrator runs; body uses FastAPI's standard validation-error format, not the plmMSA error envelope above | n/a                                                 |
| 502  | `E_INTERNAL`      | upstream embedding / align service returned an unexpected response shape or was unreachable                                                                                                                                    | `service`, `cause`, `expected`, `got`               |
| 503  | `E_GPU_OOM`       | embedding service OOM'd while encoding (very large input or insufficient VRAM)                                                                                                                                                 | `model`, `cause`                                    |

The full set of stable error codes lives in
[`src/plmmsa/errors.py`](../src/plmmsa/errors.py). Codes never get
silently renamed; new codes added to that enum are additive.

---

## 6. Examples

### 6.1 Successful request — curl

```bash
curl -sS -X POST "$BASE_URL/v2/templates/realign" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "query_sequence": "MKTIIALSY...",
  "a3m": ">7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1\nSPRQ...\n>...\n",
  "sort_by_score": true
}
JSON
```

For long A3M bodies, build the JSON with `jq` so newlines round-trip:

```bash
jq -n --arg q "$(grep -v '^>' query.fasta | tr -d '\n')" \
      --arg a "$(cat templates.a3m)" \
      '{query_sequence:$q, a3m:$a, sort_by_score:true}' \
| curl -sS -X POST "$BASE_URL/v2/templates/realign" \
       -H "Authorization: Bearer $TOKEN" \
       -H "Content-Type: application/json" \
       -d @- > response.json
```

Response payload extraction:

```bash
jq -r .payload < response.json > realigned.a3m
jq    .stats   < response.json
```

### 6.2 Successful request — Python `requests`

```python
import os
import requests

with open("query.fasta") as f:
    next(f)                         # drop the FASTA header line
    query_seq = f.read().replace("\n", "")

a3m_text = open("templates.a3m").read()

resp = requests.post(
    f"{os.environ['BASE_URL']}/v2/templates/realign",
    headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
    json={
        "query_sequence": query_seq,
        "a3m": a3m_text,
        "sort_by_score": True,
    },
    timeout=900,
)
resp.raise_for_status()
body = resp.json()

print(body["stats"])
with open("realigned.a3m", "w") as f:
    f.write(body["payload"])
```

### 6.3 Error response

```bash
curl -sS -X POST "$BASE_URL/v2/templates/realign" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query_sequence":"ABC","a3m":">t/1-5\nABCDE\n"}'
```

Response (HTTP 400):

```json
{
  "code": "E_INVALID_FASTA",
  "message": "query_sequence length does not match the a3m's match-state count.",
  "detail": {
    "query_length": 3,
    "first_drop_reason": "query_len_mismatch:5vs3"
  }
}
```

Recommended client logic:

```python
if not resp.ok:
    err = resp.json()
    if err["code"] == "E_INVALID_FASTA":
        # caller paired the wrong query — surface the parser hint
        raise BadInput(err["detail"].get("first_drop_reason", err["message"]))
    if err["code"] == "E_QUEUE_FULL":
        raise TooManyTemplates(err["detail"]["records"])
    if err["code"] in ("E_AUTH_MISSING", "E_AUTH_INVALID"):
        raise AuthError(err["code"])
    if err["code"] == "E_GPU_OOM":
        # transient — try a smaller batch later
        raise RetryableError(err["code"])
    # Unrecognized code: log and bail with the message.
    raise RuntimeError(f"plmMSA error {err['code']}: {err['message']}")
```

---

## 7. Determinism

OTalign's Sinkhorn solver is deterministic on a given device + tensor
layout. Same inputs → same `payload` byte-for-byte across reruns
**within the same plmMSA build**. Two caveats:

- Different GPU types (Ampere vs. Ada vs. Hopper) may produce
  bit-different floats due to non-associativity of float math; the
  alignment columns are typically identical but trailing decimals of
  scores can drift.
- New plmMSA releases may change OTalign hyperparameters or the
  embedding-side tokenization. Cross-version regenerations should be
  expected to differ within an identity band, not byte-for-byte. For
  reproducibility, pin a plmMSA version and stamp it on every result
  client-side.

---

## 8. Versioning

This endpoint is on the **`/v2/` surface**: URL-namespaced, no silent
breaking changes. Concretely:

- New optional request fields may be added at any time. Clients that
  don't send them get the documented default; clients that do see
  new behavior gated by the field.
- The set of stable error codes (the `code` strings) is additive —
  existing codes never get renamed. New codes are added to
  [`src/plmmsa/errors.py`](../src/plmmsa/errors.py); clients that
  branch on unknown codes should fall through to a generic-error
  branch as in [§6.3](#63-error-response).
- The output A3M format invariants ([§4.2](#42-output-a3m-format))
  are part of the contract. Changes that would violate them (e.g.
  reintroducing lowercase insertions in the output) ship under a new
  endpoint or behind an opt-in request field.
- The `stats` object is **append-only** — fields may be added but
  existing fields don't change semantics. Clients should ignore
  unknown keys.

When `/v2/` reaches end-of-life, it returns `410 E_GONE` with a
sunset date and a pointer to the successor surface. Today there is
no announced sunset.

---

## 9. Limits + sizing

| limit                                              | default | tunable                                   |
| -------------------------------------------------- | ------: | ----------------------------------------- |
| max query residues                                 |    1022 | `TemplatesRealignConfig.max_query_length` |
| max template residues per record                   |    1022 | same                                      |
| max records per request                            |    5000 | `TemplatesRealignConfig.max_records`      |
| request timeout (server-side, embed + align total) |   900 s | `PLMMSA_TEMPLATES_TIMEOUT_S` env var      |

Embedding is the bottleneck. Identical template residue strings are
de-duplicated server-side before embedding, so requests with many
PDB-chain duplicates of the same sequence run much faster than the
record count suggests. Watch `stats.unique_template_seqs` to see the
dedup payoff.

---

## 10. See also

- [`templates-realign.md`](./templates-realign.md) — operator
  walkthrough, deployment notes, design rationale.
- [`PLAN_TEMPLATES_REALIGN.md`](../PLAN_TEMPLATES_REALIGN.md) —
  full design plan including mode-default empirical comparison,
  test layout, and trade-off discussion.
- [`submitting-msa.md`](./submitting-msa.md) — sister endpoint
  `/v2/msa` for *generating* an MSA from scratch (vs. re-aligning
  an existing one).
- `src/plmmsa/errors.py` — full enumeration of stable error codes.
- `/openapi.json` on a running api service — machine-readable schema.
