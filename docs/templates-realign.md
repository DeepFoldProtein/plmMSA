# Templates re-alignment

Re-align an existing **hmmsearch-style A3M** against a query under
OTalign / Ankh-Large / glocal. Use this when you already have a
candidate set of structural templates (typically from an HMM scan
against PDB) and want PLM-driven column placements rather than HMMER's.

The endpoint is `POST /v2/templates/realign` on the api service. It is
**bearer-token gated** (same auth model as `/v2/embed` and
`/v2/align`) and currently **synchronous** — large inputs may take a
few minutes; submit-then-poll is on the roadmap.

For the design rationale (mode choice, drop-insertions rule,
header-interval semantics) see
[`PLAN_TEMPLATES_REALIGN.md`](../PLAN_TEMPLATES_REALIGN.md).

## Request

```json
POST /v2/templates/realign
Authorization: Bearer <token>
Content-Type: application/json

{
  "query_id": "T1104",
  "query_sequence": "MAQ...",
  "a3m": "<full hmmsearch a3m text>",
  "model": "ankh_large",
  "mode": "glocal",
  "options": {}
}
```

| field            | required | default       | notes                                                                                |
|------------------|----------|---------------|--------------------------------------------------------------------------------------|
| `query_id`       | no       | `"query"`     | A3M label for the query record at the top of the output.                             |
| `query_sequence` | yes      | —             | Residues. Server normalizes (uppercase, gap-strip, whitespace-strip).                |
| `a3m`            | yes      | —             | hmmsearch-style A3M body. Must satisfy `upper+gap == len(query_sequence)` per record. |
| `model`          | no       | `ankh_large`  | PLM backend id.                                                                      |
| `mode`           | no       | `glocal`      | OTalign DP mode. `q2t` / `local` / `global` / `t2q` are also accepted.               |
| `options`        | no       | `{}`          | Extra OTalign tunables passed straight through (e.g. `eps`, `tau`, `n_iter`).        |
| `sort_by_score`  | no       | `false`       | If true, output records are emitted in OTalign-score-descending order (best hit first). Default preserves input order so the output diffs cleanly against the original a3m row-by-row. |

## Response

```json
200 OK
{
  "format": "a3m",
  "payload": ">7sch_A/55-680 Score=0.875 [subseq from] mol:protein length:720  Exostosin-1\nSPRQ...\n>...",
  "stats": {
    "pipeline": "templates_realign",
    "query_length": 649,
    "records_in": 593,
    "records_kept": 593,
    "records_dropped_sanity": 0,
    "records_dropped_no_match": 0,
    "unique_template_seqs": 93,
    "model": "ankh_large",
    "mode": "glocal",
    "aligner": "otalign"
  }
}
```

The output A3M follows three invariants downstream tools can rely on:

- **The output mirrors the hmmsearch input shape** — just the template
  records, re-rendered. No query record at the top (the input doesn't
  have one either; the endpoint takes the query as a separate field).
  If your downstream tool expects a ColabFold/AlphaFold-style A3M with
  the query at index 0, prepend it client-side.
- **Every row is exactly `query_length` characters** drawn from
  `[A-Z-]`. No lowercase A3M insertions appear in the output — template
  residues that OTalign couldn't place at a query column are dropped
  (PLAN §2). Use [`POST /v2/msa`](submitting-msa.md) instead if you
  need lowercase-insert preservation.
- **Every header carries `Score=...`** in the canonical `{:.3f}`
  format adjacent to a re-intervalled `/start-end`. The domain id and
  every other tail token (`mol:protein`, `length:N`, free-text
  description) survive byte-for-byte.

## Errors

Stable codes (full list in [error taxonomy](../CLAUDE.md)):

| HTTP | code               | when                                                                  |
|------|--------------------|------------------------------------------------------------------------|
| 400  | `E_INVALID_FASTA`  | empty / non-amino-acid query, or `len(query_sequence)` ≠ a3m's match-state count. |
| 400  | `E_SEQ_TOO_LONG`   | query or any template residue count exceeds `max_query_length` (default 1022). |
| 401  | `E_AUTH_MISSING`   | `Authorization` header absent.                                         |
| 401  | `E_AUTH_INVALID`   | bearer token unknown or revoked.                                       |
| 413  | `E_QUEUE_FULL`     | a3m has more records than `TemplatesRealignConfig.max_records` (default 5000). |
| 422  | (FastAPI default)  | request body fails Pydantic validation (missing required fields).      |
| 502  | `E_INTERNAL`       | upstream embedding / align service returned an unexpected shape.       |

## Examples

### curl

```bash
curl -sS -X POST http://localhost:8080/v2/templates/realign \
  -H "Authorization: Bearer $PLMMSA_TOKEN" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON' | jq -r .payload > realigned.a3m
{
  "query_id": "T1104",
  "query_sequence": "MAQAAEKLQ...",
  "a3m": "<paste a3m here as a JSON string, or build via jq -Rs>"
}
JSON
```

When the a3m body is long, build the request with `jq` so newlines
survive the JSON round-trip:

```bash
jq -n --arg q "$(cat query.fasta | tail -n +2 | tr -d '\n')" \
      --arg a "$(cat templates.a3m)" \
      '{query_id:"T1104", query_sequence:$q, a3m:$a}' \
| curl -sS -X POST http://localhost:8080/v2/templates/realign \
       -H "Authorization: Bearer $PLMMSA_TOKEN" \
       -H "Content-Type: application/json" \
       -d @- | jq -r .payload > realigned.a3m
```

### Python (`requests`)

```python
import requests

with open("query.fasta") as f:
    next(f)  # skip header
    query_seq = f.read().replace("\n", "")

a3m_text = open("templates.a3m").read()

resp = requests.post(
    "http://localhost:8080/v2/templates/realign",
    headers={"Authorization": f"Bearer {os.environ['PLMMSA_TOKEN']}"},
    json={
        "query_id": "T1104",
        "query_sequence": query_seq,
        "a3m": a3m_text,
    },
    timeout=900,
)
resp.raise_for_status()
body = resp.json()

print(body["stats"])  # records_kept / records_dropped_* / etc.
with open("realigned.a3m", "w") as f:
    f.write(body["payload"])
```

## Operational notes

- **Embedding cost is the bottleneck.** The orchestrator deduplicates
  identical template residue strings before calling `/embed/bin`, then
  fans the unique embeddings back out per record at the align stage.
  On the bundled CASP-style fixture (593 records) this collapses to
  93 unique embeddings — a 6.4× saving. If you observe unexpectedly
  long runtimes, log the `unique_template_seqs` stat to see whether
  dedup is helping.
- **`mode` choice.** The default is `glocal`. PLAN §6.2's empirical
  comparison showed q2t loses ~25 pp on hmmsearch agreement to its
  forced-query-end artifact; only override `mode` if you have a
  specific need (e.g. forcing every query position to be informed by
  a template residue, in which case `q2t` or `t2q` may help).
- **Identity is lower than HMMER's by design.** OTalign re-places
  template residues based on PLM similarity, not HMM match-state
  posteriors; on this fixture median identity-at-matched-columns is
  ~15%, in line with §6.6 baseline. This is not a quality regression —
  it reflects what the PLM actually sees.

## Sample output

[`bin/realign_fixture.py`](../bin/realign_fixture.py) runs the
orchestrator end-to-end on the bundled fixture and writes the result
to `tmp/exostosin_realigned.a3m` (gitignored — reproduce locally):

```bash
PLMMSA_TEST_EMBEDDING_URL=http://172.28.0.3:8081 \
PLMMSA_TEST_ALIGN_URL=http://172.28.0.7:8083    \
    uv run python bin/realign_fixture.py
```

On a single Ada GPU the 593-record fixture re-aligns in ~12 s. All
records survive (the no-match drop is 0 on this fixture, well below
the 10% floor §6.6 enforces). Records that share residues with the
query get the top score (`6.398` for the five PDB chains identical
to the query); a typical distant-homolog record (e.g. `1omx_A`,
"Alpha-1,4-N-acetylhexosaminyltransferase EXTL2") lands around
`Score=3.99`.
