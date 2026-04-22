# Warming PLM weights

Step 2 of [`PLAN.md`](../PLAN.md)'s service-readiness checklist. Populates
`MODEL_CACHE_DIR` with the HuggingFace snapshots the `embedding` service
reads at startup. Once warm, the container runs offline — no runtime
HF downloads, no outbound traffic from the embedding worker.

## Why a separate "warming" step

The `embedding` container mounts `MODEL_CACHE_DIR` **read-only** by default
(see `docker-compose.yml`) and runs with `HF_HUB_OFFLINE=1`. That's
deliberate: a GPU worker that can't phone home is easier to reason about,
and `/store/deepfold/huggingface` (our shared cache) should not be
writable from a per-service container. So warming happens **from the
host**, once, and the container inherits the warm tree.

## Prereqs

- `MODEL_CACHE_DIR` set in `.env` to the host dir that will hold the cache.
  The deepfold host's shared warm tree is `/store/deepfold/huggingface`;
  point at it unless you have a reason not to.
- `uv sync --all-extras` on the host — pulls `huggingface-hub` (and the
  `hf` / `huggingface-cli` commands).
- Network reachability to `huggingface.co` from the host (not from the
  container).

## The `hf` CLI

`huggingface-hub >= 0.30` ships the short `hf` command; older versions use
`huggingface-cli`. Both syntaxes work. Every example below is equivalent
with `huggingface-cli download` substituted for `hf download`.

```bash
# Verify what's available in your venv:
uv run hf --version               # hf CLI (preferred if >= 0.30)
uv run huggingface-cli --version  # longer form, always available
```

## Download commands

All commands write into `$HF_HOME/hub/models--<org>--<repo>/`. Set
`HF_HOME` to the same value `MODEL_CACHE_DIR` resolves to in `.env` so the
host-side warm tree matches what the container sees at
`/model_cache/hub/...`.

```bash
export HF_HOME="$(grep '^MODEL_CACHE_DIR=' .env | cut -d= -f2-)"
```

### Ankh-CL (required — this is the `ankh_cl` backend checkpoint)

```bash
uv run hf download DeepFoldProtein/Ankh-Large-Contrastive
```

Approx 4.4 GB.

### Ankh-Large (tokenizer + full encoder)

```bash
uv run hf download ElnaggarLab/ankh-large
```

Approx 15 GB. Required even if `[models.ankh_large].enabled = false`, because
`ankh_cl` uses the same tokenizer.

### ESM-1b

```bash
uv run hf download facebook/esm1b_t33_650M_UR50S
```

Approx 4.9 GB.

### ProtT5-XL-UniRef50 (the biggest)

```bash
uv run hf download Rostlab/prot_t5_xl_uniref50
```

Approx 22 GB. Skip if you don't plan to enable `prott5`.

### Batch (all four)

```bash
for repo in \
    DeepFoldProtein/Ankh-Large-Contrastive \
    ElnaggarLab/ankh-large \
    facebook/esm1b_t33_650M_UR50S \
    Rostlab/prot_t5_xl_uniref50 ; do
    uv run hf download "$repo"
done
```

## Verify

```bash
du -sh "$HF_HOME"/hub/models--*
```

Every enabled model should show a directory under `hub/models--<org>--<repo>/`
with a populated `snapshots/<revision>/` containing the weight files
(`pytorch_model.bin` or `model.safetensors`, plus `config.json`,
`tokenizer.json`, etc.).

## Which PLMs to enable

Toggle in `settings.toml` under `[models.<name>].enabled`. Each enabled
backend is loaded at `embedding` startup — disable the ones you don't
need to cut boot time and GPU footprint.

| Backend     | Checkpoint | GPU load | Default device env |
| ----------- | ---------- | -------- | ------------------ |
| `ankh_cl`   | ~4.4 GB    | ~6–8 GB  | `ANKH_CL_DEVICE`   |
| `ankh_large`| ~15 GB     | ~16–20 GB| `ANKH_LARGE_DEVICE`|
| `esm1b`     | ~4.9 GB    | ~5–7 GB  | `ESM1B_DEVICE`     |
| `prott5`    | ~22 GB     | ~22–26 GB| `PROTTRANS_DEVICE` |

On a 2×48 GB GPU host, all four fit comfortably when split across both
cards. A sensible split for 2-GPU hosts:

```env
ANKH_CL_DEVICE=cuda:0
ANKH_LARGE_DEVICE=cuda:0
ESM1B_DEVICE=cuda:1
PROTTRANS_DEVICE=cuda:1
```

## Bring the service up offline

`docker-compose.yml` already pins `HF_HUB_OFFLINE=1` and the RO mount for
the embedding container — no extra steps.

```bash
docker compose up -d embedding
docker compose exec embedding curl -s http://localhost:8081/health | jq
```

Expect each enabled backend to appear under `models` with
`loaded: true, device: "cuda:N", dim: <D>`.

## Adding a new PLM

1. Implement a `PLM` subclass in `src/plmmsa/plm/<name>.py` (see `plm/ankh_cl.py`).
2. Register it in `src/plmmsa/plm/registry.py`.
3. Add a `[models.<name>]` block in `settings.toml`.
4. Warm its checkpoint with `uv run hf download <repo>`.
5. Restart the embedding service.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: [Errno 30] Read-only file system` on download attempt | `HF_HUB_OFFLINE=0` but mount is RO | Warm from host first, leave `HF_HUB_OFFLINE=1` |
| Health shows a backend missing from `models` | Disabled in `settings.toml`, or `_load_*` threw (see `docker compose logs embedding`) | Re-enable + confirm checkpoint exists under `$HF_HOME/hub/` |
| First `/embed` call is slow | First-pass CUDA kernel compilation | Warmup by calling `/embed` with a short sequence right after startup |
| "Tokenizer not found" on Ankh-CL | `ElnaggarLab/ankh-large` tokenizer not cached | `hf download ElnaggarLab/ankh-large` (Ankh-CL reuses the Ankh-Large tokenizer) |
