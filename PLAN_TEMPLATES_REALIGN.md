# PLAN — OTalign template re-aligner

Take an existing **hmmsearch-style A3M** (templates already snapped to query
match columns by HMMER) and **re-align each template against the query with
OTalign**, producing the same-shape A3M with re-computed columns and an
alignment score on every header. Default PLM: **Ankh-Large**. Default DP
mode: **glocal**.

> **Mode default — empirically validated.** First draft pinned `q2t` per
> upstream convention. The §6.2 head-to-head on a 20-record fixture sample
> (1903 template residues) showed glocal beats q2t by ~25 pp on hmmsearch
> agreement and ~9 pp on template coverage — q2t's forced
> `query_end == Lq` artifact pulls the trailing template residue onto the
> wrong query column. glocal (free end-gaps both sides) is the right
> default; q2t stays callable as an opt-in mode.
>
> | mode    | coverage | hmmagree | identity |
> |---------|---------:|---------:|---------:|
> | q2t     |    0.854 |    0.349 |    0.127 |
> | glocal  |    0.949 |    0.595 |    0.152 |
> | local   |    0.938 |    0.595 |    0.155 |

Sample fixture: [tests/data/templates_realign/exostosin_hmmsearch.a3m](tests/data/templates_realign/exostosin_hmmsearch.a3m)
(593 records, query length 649, template lengths 26..733).

---

## 1. Input shape & sanity rules

### 1.1 Query normalization

The caller's `query_sequence` is **not trusted** to be in canonical form.
First step on entry, before *any* length check or model call:

```python
query_seq = "".join(query_sequence.upper().split()).replace("-", "")
```

Specifically: uppercase everything (lowercase = insertion-state in A3M
parlance and is meaningless on a free-standing query), strip whitespace,
strip gap characters. After normalization, validate alphabet against
`A..Z`. Anything still non-alphabetic = `E_INVALID_FASTA`.

The fixture's intended query is

```
SPRQKRDANSSIYKGKKCRMESCFDFTLCKKNGFKVYVYPQQKGEKIAESYQNILAAIEG
SRFYTSDPSQACLFVLSLDTLDRDQLSPQYVHNLRSKVQSLHLWNNGRNHLIFNLYSGTW
PDYTEDVGFDIGQAMLAKASISTENFRPNFDVSIPLFSKDHPRTGGERGFLKFNTIPPLR
KYMLVFKGKRYLTGIGSDTRNALYHVHNGEDVVLLTTCKHGKDWQKHKDSRCDRDNTEYE
KYDYREMLHNATFCLVPRGRRLGSFRFLEALQAACVPVMLSNGWELPFSEVINWNQAAVI
GDERLLLQIPSTIRSIHQDKILALRQQTQFLWEAYFSSVEKIVLTTLEIIQDRIFKHISR
NSLIWNKHPGGLFVLPQYSSYLGDFPYYYANLGLKPPSKFTAVIHAVTPLVSQSQPVLKL
LVAAAKSQYCAQIIVLWNCDKPLPAKHRWPATAVPVVVIEGESKVMSSRFLPYDNIITDA
VLSLDEDTVLSTTEVDFAFTVWQSFPERIVGYPARSHFWDNSKERWGYTSKWTNDYSMVL
TGAAIYHKYYHYLYSHYLPASLKNMVDQLANCEDILMNFLVSAVTKLPPIKVTQKKQYKE
TMMGQTSRASRWADPDHFAQRQSCMNTFASWFGYMPLIHSQMRLDPVLF
```

After normalization: `len(query_seq) == 649`, equals the `upper+gap` count
of every record in the fixture, and equals row 0 (`7sch_A/55-703`) verbatim
once that row's lowercase/gaps are stripped. We use this as the canonical
query in §6.

### 1.2 Per-record sanity rules

For every record `>id/start-end ...` followed by an A3M row:

- `upper + gap == query_len` — the row carries `query_len` match-state slots,
  gap-padded.
- `upper + lower == end - start + 1` — non-gap residues equal the header's
  interval length (the actual template residues we'll feed to the PLM).
- alphabet: `A..Z`, `a..z`, `-` only; anything else is an alphabet error.

Verified on the fixture: every one of 593 records satisfies both rules
(`upper+gap = 649`; `upper+lower = end-start+1`; no stray characters). See
`scripts/sanity_check_hmmsearch_a3m.py` (new, §6).

### 1.3 Query / file cross-check

The query is **always** taken from the caller's `query_sequence` field; we
never infer it from the A3M body. Record 0 is treated like any other
template — its alignment gets re-computed against the query and emitted in
the output. The fixture happens to seed record 0 with `7sch_A/55-703` whose
residues equal the cleaned query (a coincidence of how UniRef-seeded
hmmsearch builds its alignment), but that is **not** a contract — other
hmmsearch dumps will have an unrelated record 0 and we must handle them
identically.

The one cross-check we *do* enforce is structural: `len(query_seq)` must
equal the per-row `upper+gap` count (which is constant across the file by
A3M definition). Mismatch ⇒ `E_INVALID_FASTA` — the caller paired the wrong
query with this A3M. As a soft signal we log at INFO when record 0's
normalized residues equal `query_seq` (positive confirmation when present);
silence is not an error.

The legacy hmmsearch dump has no separate query header, so we expect
`(query_seq, a3m)` as **two separate inputs**.

---

## 2. End-to-end flow

```
(query_sequence_input, a3m_text)
  │
  ├─ normalize query (§1.1) → query_seq = upper().strip_ws().replace("-", "")
  │     • assert len(query_seq) == upper+gap of every record (file-level
  │       invariant; mismatch = E_INVALID_FASTA, see §1.3).
  │
  ├─ parse → records = [(header, raw_seq, interval, a3m_row), ...]
  │     • raw_seq = "".join(c.upper() for c in a3m_row if c != "-")
  │     • drop records that fail §1 sanity rules (counted in stats)
  │
  ├─ enforce limits (§5)
  │     • len(query_seq) ≤ max_length(ankh_large)  → 1022
  │     • max(len(t.raw_seq)) ≤ max_length         → 1022
  │     • len(records) ≤ settings.templates.max_records (default 5_000)
  │
  ├─ dedupe templates by raw_seq → unique_seqs[]
  │     • fixture has ~30% dup rate; pays the embed cost once per unique seq.
  │
  ├─ embed via the existing /embed/bin path
  │     • POST {model: "ankh_large", sequences: [query_seq, *unique_seqs]}
  │     • re-uses plmmsa.pipeline.orchestrator._embed_chunks (binary frame).
  │     • length-sorted batching keeps padding minimal.
  │
  ├─ align via /align/bin (one batched call, OTalign)
  │     • aligner="otalign", mode="glocal" (see mode-default note above)
  │     • options inherited from settings.aligners.otalign defaults
  │       (eps, tau, n_iter, tol, gap factors, fused_sinkhorn, device).
  │
  ├─ filter out query-side insertions
  │     • OTalign's `(qi=-1, ti>=0)` columns are template residues with
  │       no matching query position. In standard A3M they would render
  │       as lowercase letters between match-state slots; this pipeline
  │       does NOT emit lowercase. **Drop those columns** (trim the
  │       offending template residues from the output).
  │     • Surviving columns are exclusively `(qi>=0, ti>=0)` (match) or
  │       `(qi>=0, ti=-1)` (gap in target). The rendered row therefore
  │       contains only `[A-Z]` and `-` and is exactly `query_len` chars.
  │     • Trade-off: a template residue that OTalign couldn't place at a
  │       query column is silently dropped from the output. With glocal
  │       on this fixture the drop rate is ~5% (coverage 0.949). Dropped
  │       residues do not appear anywhere in the row — A3M consumers
  │       that care about every template residue should use a different
  │       endpoint (TBD).
  │
  ├─ map filtered alignment back onto records
  │     • column list (qi, ti) is in raw_seq coordinates → render via a
  │       new helper `render_hit_match_only(query_len, hit)` that emits
  │       only match-state slots (no lowercase between them). Distinct
  │       from `plmmsa.pipeline.a3m.render_hit` which keeps inserts.
  │     • Output row invariant: `len(row) == query_len`,
  │       `row.replace("-", "").isupper()` is True.
  │     • Re-interval the header to span the kept template residues:
  │           kept_ti = sorted({ti for qi,ti in cols if qi>=0 and ti>=0})
  │           new_start = orig_start + kept_ti[0]              # 1-based
  │           new_end   = orig_start + kept_ti[-1]             # 1-based
  │       For glocal alignments where every template residue gets placed
  │       (~95% of records on this fixture), `new_start..new_end` is the
  │       contiguous slice OTalign actually used. When interior residues
  │       are dropped (~5% of records), `new_end - new_start + 1` exceeds
  │       the kept-residue count — the interval is the *span*, not the
  │       residue count. We accept this loss of fidelity for the
  │       simplicity of a single A3M-compatible header.
  │
  └─ assemble output A3M
        • header preserved verbatim except for two surgical edits:
              (1) `/start-end` → `/new_start-new_end` (re-intervalled).
              (2) `Score=…` token: stripped if present, then re-stamped at
                  the end with the new value.
          Everything else stays — the **domain name / id** (`7sch_A`),
          the `[subseq from]` annotation, `mol:protein`, the `length:N`
          token (full-protein length from the PDB header), the
          human-readable description (`Exostosin-1`), and any other
          provenance tokens hmmsearch wrote. Downstream tools that key
          on the domain name or description don't see a difference.
        • A3M-row length invariant. `render_hit` always emits exactly
          `query_len` match-state slots: any query column that OTalign's
          path didn't visit is represented as `-` (gap) in that slot. So
          `upper+gap == query_len` holds for every output row, even when
          q2t leaves query end-residues unaligned. No row needs manual
          padding past what `render_hit` already produces, but the
          invariant is explicitly tested (§6.4 / §6.5 / §6.6).
        • query record prepended:
              >{query_id} Score={query_self_score:.3f}
              {query_seq}
        • query_self_score = OTalign(query, query) score from the same batch.
```

OTalign scoring topology: identical PLM for retrieval and scoring (Ankh-Large
on both sides). Cross-PLM is overkill here — there's no retrieval step, and
the upstream OTalign default uses Ankh-Large for both.

---

## 3. API surface

New endpoint on the **api** service:

```
POST /v2/templates/realign
GET  /v2/templates/realign/{job_id}
DELETE /v2/templates/realign/{job_id}
```

Request:

```json
{
  "query_id": "T1104",          // optional; default "query"
  "query_sequence": "MAQ...",   // required, ≤ max_length
  "a3m": "<hmmsearch a3m text>",// required, multi-record A3M body
  "model": "ankh_large",         // optional; pinned default "ankh_large"
  "mode": "glocal",              // optional; pinned default "glocal"
  "options": {                   // optional; passes through to OTalign
    "eps": 0.1, "tau": 1.0, "n_iter": 1250,
    "fused_sinkhorn": false
  }
}
```

Response (sync small inputs OR job-poll for large):

```json
{
  "format": "a3m",
  "payload": "<re-aligned a3m text>",
  "stats": {
    "pipeline": "templates_realign",
    "query_length": 649,
    "records_in": 593,
    "records_kept": 593,
    "records_dropped_sanity": 0,
    "records_dropped_oob": 0,
    "unique_template_seqs": 412,
    "model": "ankh_large",
    "mode": "q2t",
    "aligner": "otalign",
    "query_self_score": 1.234
  }
}
```

**Job lifecycle.** Submission threshold: if `records_in × max_template_len >
settings.templates.sync_budget` (proposed default `5e6`), enqueue and return
`202` with a `job_id`; otherwise compute synchronously and return `200`. The
async path reuses the existing `JobStore` / worker pattern (same wire shape as
`/v2/msa`).

**Auth.** Same as `/v2/embed` — `Depends(require_admin_token)` until we
decide the public abuse profile. Cheap to relax later.

---

## 4. Module layout

New code lives under `src/plmmsa/templates/`:

```
src/plmmsa/templates/
├── __init__.py
├── a3m_parser.py        # parse_hmmsearch_a3m(text) -> [Record]; sanity validators.
├── pipeline.py          # TemplatesRealignOrchestrator: embed → align → re-render.
└── render.py            # rebuild a3m text from (query, records, alignments).
```

Wire-up:

- `src/plmmsa/api/routes/v2.py` — add the three routes (`POST /v2/templates/realign`
  + `GET` / `DELETE`). Pattern after the existing `/v2/msa` handlers; no new
  file.
- `src/plmmsa/worker/__main__.py` — register `templates_realign` job kind so
  the async path uses the same worker process.
- `src/plmmsa/pipeline/__init__.py` — re-export `parse_hmmsearch_a3m` and
  `Record` so tests / bench scripts can import without grabbing the full
  pipeline module.

No changes needed to `align/`, `embedding/`, `vdb/`, or service Dockerfiles.
The new code is api-side orchestration over endpoints we already ship.

---

## 5. Limits, error model, validation

| Condition                                            | Error code           | HTTP |
|------------------------------------------------------|----------------------|------|
| `len(query_sequence) > max_length(ankh_large)`       | `E_SEQ_TOO_LONG`     | 400  |
| any template residue count > max_length              | `E_SEQ_TOO_LONG`     | 400  |
| `len(records) > settings.templates.max_records`      | `E_QUEUE_FULL`       | 413  |
| record fails sanity (`upper+gap != query_len`, etc.) | counted, not raised  | —    |
| zero records survive sanity                          | `E_INVALID_FASTA`    | 400  |
| Ankh-Large not loaded on `embedding`                 | `E_UNSUPPORTED_MODEL`| 400  |
| GPU OOM during embed/align                           | `E_GPU_OOM`          | 503  |
| job cancelled mid-flight                             | `E_CANCELLED`        | 499  |

Sanity-rule failures are non-fatal per record — bad records are dropped, the
count surfaces in `stats.records_dropped_sanity`. Whole-job failure only when
the input is structurally unparseable.

A `tools.realign_sanity` script (§6) lets operators pre-validate an A3M
locally without hitting the server.

---

## 6. Tests

Everything lives under `tests/`. Layered from cheapest to slowest, and
ordered so that **OTalign's contract is pinned down before any pipeline
glue is written** — otherwise §6.4 onwards would be testing parser +
renderer + scorer against a moving target.

```
tests/
├── data/templates_realign/
│   └── exostosin_hmmsearch.a3m    # was tmp/templates_hmmsearch.a3m
├── test_otalign_behavior.py             # §6.1 — synthetic embeddings, no model
├── test_otalign_real_embeddings.py      # §6.2 — RUN_SLOW=1, Ankh-Large
├── test_templates_a3m_sanity.py         # §6.3 — pure-data
├── test_templates_a3m_roundtrip.py      # §6.4 — pure-data
├── test_templates_pipeline_unit.py      # §6.5 — pure-data, stubbed services
└── test_templates_realign_consistency.py# §6.6 — RUN_SLOW=1, end-to-end
```

Fast tests (§6.1, §6.3, §6.4, §6.5) are part of the default `uv run pytest`
set. Slow tests (§6.2, §6.6) ride under
`@pytest.mark.skipif(not os.environ.get("RUN_SLOW"))`, matching the
existing PLM integration tests.

### 6.1 OTalign behavior — synthetic embeddings (no model load)

`tests/test_otalign_behavior.py`. Drives `OTalign().align(...)` directly
with hand-built float32 arrays so we exercise the DP + Sinkhorn contract
without any PLM cost. **These tests are the prerequisite for everything
downstream** — if q2t doesn't behave the way the pipeline assumes, fixing
the pipeline can't help.

| #  | Scenario                                                     | Asserts                                                                                                       |
|----|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| 1  | `q == t` (same matrix), q2t                                  | `columns == [(i, i) for i in range(L)]`; `score > 0`; `target_start=0`, `target_end=L-1`.                     |
| 2  | `t = q[a:b]` substring, q2t                                  | `target_start=0`, `target_end=b-a-1`; `query_start=a`, `query_end=b-1`; identity = 1.0 on matched columns.    |
| 3  | `t = q[a:b] + noise[c:d]` (template extends beyond query)    | template-global preserved: every template index appears in `columns`; query end-gaps free (no penalty band).  |
| 4  | Permuted noise template                                      | runs to completion; no out-of-bounds columns; score noticeably lower than scenario 1.                         |
| 5  | Mode contract — q2t / glocal / local / global / t2q          | q2t: every template index used; t2q: every query index used; local: no end-gap regions; global: no trimming. |
| 6  | Determinism                                                  | Two identical calls (same seed via `numpy`) → identical columns + score (within `tol`).                       |
| 7  | Degenerate sizes — Lq=1, Lt=1; Lq=1022, Lt=10                | runs without exception; columns shape matches DP mode contract.                                               |
| 8  | Column bounds invariant                                      | every `(qi, ti)` satisfies `qi ∈ {-1, 0..Lq-1}`, `ti ∈ {-1, 0..Lt-1}`; never both `-1`.                        |
| 9  | `target_start/target_end` map to actually-used template idxs | the smallest / largest `ti` in `columns` (excluding -1) equal `target_start` / `target_end`.                  |

Tests use small embedding dims (e.g. `D=32`) and short sequences (`L≤64`)
so the whole file finishes in seconds. We're testing structure, not
biology.

### 6.2 OTalign behavior — real Ankh-Large embeddings (`RUN_SLOW=1`)

`tests/test_otalign_real_embeddings.py`. Loads Ankh-Large in-process (the
shared HF cache at `/gpfs/deepfold/model_cache` — see CLAUDE.md memory) and
runs OTalign on actual embeddings.

- **Self-alignment of the fixture query** (the 649-aa Exostosin sequence):
  q2t should produce the diagonal; identity ≥ 0.99; score sets the upper
  bound for the score-scale calibration.
- **Record 0 of the fixture** (`7sch_A/55-703` — happens to equal the query
  for this fixture; coincidence, see §1.3) re-aligned in q2t: diagonal,
  identity ≥ 0.99.
- **A short subseq template** (e.g. query[100:200] embedded directly): q2t
  recovers `target_start=0..target_end=99`, `query_start=100..query_end=199`.
- **Score histogram** over a 20-record sampled subset: median score must
  fall in the band that makes `[aligners.otalign].filter_threshold = 0.25`
  meaningful (i.e., scores aren't all `<<0.25` or all `>>0.25`). If the
  histogram is degenerate, calibration is wrong and the §6.6 filter would
  empty the MSA.

This file is the gate before §6.6 — it confirms OTalign + Ankh-Large
*together* behave as the pipeline assumes.

### 6.3 `tests/test_templates_a3m_sanity.py` (no model load)

For the bundled fixture:

- count records, tally `(upper, lower, gap)` per record;
- assert `upper+gap` is constant across the file (= query length implied by
  the file);
- assert `upper+lower == end-start+1` for every record;
- assert alphabet is `A..Z | a..z | -` only;
- snapshot summary (n_records=593, query_len=649, len_min=26 / median=103
  / max=733) — values pinned so a fixture swap fails loudly.

Plus a handful of hand-crafted A3M strings to cover parser edge cases:
multi-line records, blank lines between records, `#` comments, leading /
trailing whitespace, malformed `>id/start-end` header.

### 6.4 `tests/test_templates_a3m_roundtrip.py` (no model load)

For every fixture record, pretend OTalign returned the *original*
hmmsearch column list (parsed from the row) and re-render with
`render_hit`. The output must equal the input row exactly. This pins down
the parser ↔ renderer contract before any PLM math gets involved.

```python
for record in records:
    cols = columns_from_a3m_row(record.row, query_len)   # new helper
    rendered = render_hit(query_len, AlignmentHit(
        target_id=record.id,
        score=0.0,
        target_seq=record.raw_seq,
        columns=cols,
    ))
    assert rendered == record.row
```

Failure here means our parser or `render_hit` disagrees on insert
placement, and any OTalign-produced row would inherit the same bug.

### 6.5 `tests/test_templates_pipeline_unit.py` — stubbed services

Pure-Python unit tests for the orchestrator, with the embedding and align
HTTP calls replaced by in-process stubs that return canned shapes. No
weights loaded, no httpx. Verifies the pieces between §6.1 (OTalign
contract) and §6.6 (real end-to-end):

- **Header re-interval** — given an `Alignment` with
  `target_start=5, target_end=200` and `orig_start=100, orig_end=400`, the
  new header reads `/105-300`. Length tail token unchanged.
- **Domain-name + tail preserved** — header
  `>7sch_A/55-703 [subseq from] mol:protein length:720  Exostosin-1`
  re-intervalled to `/55-680` and stamped with score 0.42 must come out as
  `>7sch_A/55-680 [subseq from] mol:protein length:720  Exostosin-1 Score=0.420`.
  `7sch_A`, `[subseq from]`, `mol:protein`, `length:720`, and `Exostosin-1`
  are all byte-identical. Whitespace between tail tokens preserved.
- **`Score=` stamping** — header without any score gets `Score=…` appended;
  header that already carries `Score=…` (or `score=…`) has it stripped and
  replaced (§7.2 proposed behavior). Domain name + tail still preserved.
- **Row gap-padding to `query_len`** — synthetic OTalign output that covers
  only `q ∈ [10, 50)` of a 100-residue query produces a rendered row whose
  first 10 slots and last 50 slots are `-`, with `upper+gap == 100`. No
  row is ever shorter than `query_len` in match-state slots.
- **No lowercase in the output row** — given a synthetic OTalign result with
  `(qi=-1, ti=k)` columns interleaved among match columns, the rendered
  row contains zero lowercase letters; the offending template residues are
  dropped and the kept-residue list reflects this. Pin both
  `set(row) ⊆ set("ACDEFGHIKLMNPQRSTVWY-")` and `len(row) == query_len`.
- **Length-mismatch fail mode** — `len(query_sequence)` != per-record
  `upper+gap` ⇒ `PlmMSAError(E_INVALID_FASTA, 400)` raised on parse.
- **Dedup of identical templates** — two records with identical residues
  trigger exactly one embed call (stub records call counts); both rows in
  the output get populated.
- **Sanity-failed records are dropped, job survives** — one bogus record
  out of three; stats report `records_dropped_sanity=1`,
  `records_kept=2`, output A3M has 2 hit rows.
- **Empty / single-record / one-residue templates** — runs without
  exception, output A3M well-formed, stats report sane numbers.
- **Sync vs. async cutoff** — request below `settings.templates.sync_budget`
  returns inline (`format=a3m`); request above gets enqueued (returns a
  `job_id` from the stubbed JobStore).

### 6.6 `tests/test_templates_realign_consistency.py` — `RUN_SLOW=1`

End-to-end on the fixture: re-align all 593 records with OTalign /
Ankh-Large / q2t, then assert:

- `n_dropped_oob == 0` — every alignment fits inside the template.
- every emitted row passes the §6.4 round-trip check (parser ↔ renderer
  invariant survives an OTalign-produced column list).
- per-row: `upper+gap` of the new row equals `query_len` (`render_hit`
  guarantees this; verify so a regression surfaces here too).
- per-row: `upper+lower` of the new row equals `new_end - new_start + 1`,
  where `new_start = orig_start + a.target_start` and
  `new_end = orig_start + a.target_end` (re-intervalled header, see §2).
  q2t typically keeps the full template, but the assertion is written
  against the realignment's effective span so it stays correct for any
  mode.
- median identity (`Alignment.identity()`) over hmmsearch-matched columns
  ≥ a band pinned from the §6.2 self-alignment baseline.

Also computes a side-by-side report (record-level deltas in
`upper / lower / gap`, score histogram, identity bands, count of rows
where OTalign happens to reproduce hmmsearch exactly). The report is
written to `tests/data/templates_realign/last_run_report.json` (gitignored
— only the numeric thresholds in the test file are committed). Output A3M
lives at `last_run.a3m` in the same directory for eyeballing 20 random
records.

### 6.7 Mode comparison — same test file, parametrized

`pytest.mark.parametrize("mode", ["q2t", "glocal", "local"])` over a
sampled subset (~50 records). Asserts **glocal** produces the highest
hmmsearch-agreement (the default we picked after §6.2's empirical
comparison) and that q2t agreement is at least within 30 pp of glocal
(it failed by 25 pp on our 20-record sample, so this floor catches a
deeper regression but doesn't re-litigate the default). Lives under the
same `RUN_SLOW=1` gate as §6.6.

---

## 7. Open questions for review

1. **Sync vs. async cutoff.** Proposed `5e6` (records × max_len product);
   too tight? At fixture size (593 × 733 ≈ 4.3e5) we're well below, so the
   sample request would run sync. Larger inputs need queueing.

2. **Pre-existing `Score=` tokens.** The fixture has none, but the API
   contract should say what we do if a header already carries one. Proposal:
   strip it, replace with our value. Alternative: keep both as
   `OldScore=…  Score=…` for traceability.

3. **Record 0 is not the query.** The fixture happens to have record 0 ==
   query, but that's coincidence — other hmmsearch dumps don't. We commit
   to: query always comes from `query_sequence`, record 0 is just another
   template (re-aligned and emitted in the output like any other row). The
   only file-level check is `len(query_seq) == upper+gap` per record (§1.3).
   Open: do we want a `query_sequence_optional=true` mode that *does* read
   record 0 when the caller hasn't supplied a query? Cheap to add later;
   not in v1 of the endpoint.

4. **Result caching.** Cache key candidate:
   `sha256(query_seq | model | mode | sorted(unique_template_seqs) | options_canonical)`.
   Worth turning on from day one or not? My take: yes, hit rate will be high
   on repeat CASP / Boltz pipelines.

5. **Rendering long inserts.** If OTalign chooses to gap a huge stretch of
   query and dump a 200-residue lowercase run, downstream parsers (ColabFold
   notebooks) still tolerate it, but it inflates the file. Add a soft cap
   (drop record? truncate insert? warn-only?) — TBD.

---

## 8. Done-when

- `POST /v2/templates/realign` returns a re-aligned A3M for the fixture in
  one round-trip; `GET /v2/version` lists the new pipeline.
- §6.1, §6.2, §6.3, §6.4 all pass on the fixture. Fixture A3M tracked at
  `tests/data/templates_realign/exostosin_hmmsearch.a3m`; per-run output
  artifacts (`last_run.a3m`, `last_run_report.json`) gitignored.
- `tests/test_templates_a3m_sanity.py` and
  `tests/test_templates_a3m_roundtrip.py` are part of the default `uv run
  pytest` set; `tests/test_templates_realign_consistency.py` rides under
  `RUN_SLOW=1` like the existing PLM tests.
- Operator docs: one section in `docs/operator-guide.md` covering the new
  endpoint, plus a curl + Python `requests` snippet in `docs/examples.md`.
- `PLAN.md` "Shipped" gets a new line under M? once §6 is green.
