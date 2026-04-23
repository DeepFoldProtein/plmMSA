from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

SETTINGS_FILE_ENV = "PLMMSA_SETTINGS_FILE"
DEFAULT_SETTINGS_FILE = "settings.toml"


def _resolve_settings_file() -> Path:
    return Path(os.environ.get(SETTINGS_FILE_ENV, DEFAULT_SETTINGS_FILE))


class ServiceSettings(BaseModel):
    api_port: int = 8080
    embedding_port: int = 8081
    vdb_port: int = 8082
    align_port: int = 8083
    cache_port: int = 6379


class AlignerEntry(BaseModel):
    display_name: str
    enabled: bool = True
    # Upstream PLMAlign Algorithm 1 step 5 filter — drops hits whose
    # alignment score is below `min(0.2 * len(Q), 8.0)`. Aligner-
    # specific because the threshold assumes dot-product alignment
    # scores; OTalign's transport-mass score lives on a different
    # scale so its default is off. Per-request `filter_by_score`
    # always wins over this setting.
    filter_enabled: bool = False


class PlmAlignEntry(AlignerEntry):
    # PLMAlign's dot-product alignment score is what the upstream
    # threshold was calibrated against — enable by default.
    filter_enabled: bool = True
    # Scoring mode for the similarity matrix. See PLMAlign docstring:
    #   "dot_zscore" — upstream-compatible default
    #   "cosine"     — L2-normalized dot product
    #   "dot"        — raw dot product (debug only)
    score_matrix: str = "dot_zscore"
    gap_open: float = 10.0
    gap_extend: float = 1.0
    # PLM used to build the score matrix. Empty = score with the PLM that
    # searched the VDB. Upstream PLMAlign pairs the affine-gap SW aligner
    # with ProtT5 embeddings — re-embeds query fresh, pulls targets from
    # the precomputed shard store.
    score_model: str = "prott5"
    # Device for score-matrix construction. "cpu" (default) uses the numpy
    # builders in score_matrix.py; "cuda:0" (or any torch device) swaps in
    # the torch-backed builders from torch_score_matrix.py at align-service
    # startup. No client API change — same `score_matrix` enum on the wire.
    score_matrix_device: str = "cpu"


class PlmBlastEntry(AlignerEntry):
    """pLM-BLAST aligner tunables."""

    # Same dot-product-alignment score scale as PLMAlign — same default.
    filter_enabled: bool = True
    score_matrix: str = "dot_zscore"
    # Linear gap penalty — pLM-BLAST doesn't split open/extend. 0.0 matches
    # upstream's permissive default (favors long spans with embedded gaps).
    gap_penalty: float = 0.0
    # Minimum span length (cells) to report. Upstream default 20.
    min_span: int = 20
    # Moving-average smoothing window for span extraction.
    window_size: int = 20
    # Spans are kept where the smoothed signal exceeds
    # `sigma_factor * std(sim)`. Upstream default 1.0.
    sigma_factor: float = 1.0
    # Border-seed stride. 1 seeds every edge cell; 2 halves the seed count.
    border_stride: int = 1
    # Upstream's pLM-BLAST (called "plmalign" in their code) also uses
    # ProtT5 scoring. Same contract as PlmAlignEntry.score_model above.
    score_model: str = "prott5"
    score_matrix_device: str = "cpu"


class OTAlignEntry(AlignerEntry):
    """OTalign aligner tunables."""

    # OTalign's score is sum of transport-plan mass on matched cells
    # (range ~[0, 1]); the upstream PLMAlign threshold zeroes every
    # hit on this scale, so leave the filter off until an OTalign-
    # specific calibration is dialed in.
    filter_enabled: bool = False
    # OTalign upstream (DeepFoldProtein/OTalign) pairs with Ankh-Large
    # embeddings by default; fp16 is acceptable on this model family.
    score_model: str = "ankh_large"
    # FlashSinkhorn — replace the per-iteration torch Sinkhorn with a
    # torch.compile'd fused-chunk variant. Same math (log-space UOT,
    # fp32), ~3-5x faster at the CASP15 matrix sizes. Opt-in until
    # cross-validated on the full regression fixture set.
    fused_sinkhorn: bool = False
    # Torch device for the Sinkhorn solver + cost matrix. Empty string
    # = auto (cuda if available, cpu otherwise). Pin to "cuda:0" or
    # "cpu" for reproducibility. Align service must have a GPU
    # reservation (see docker-compose.yml) for cuda to actually work.
    device: str = ""


class AlignerRegistry(BaseModel):
    plmalign: PlmAlignEntry
    plm_blast: PlmBlastEntry = PlmBlastEntry(
        display_name="pLM-BLAST (multi-path SW + span extraction)",
        enabled=False,
    )
    otalign: OTAlignEntry


class VdbCollectionSettings(BaseModel):
    display_name: str
    model_backend: str  # Which PLM produced the vectors in this index.
    dim: int
    index_path: str  # Relative to VDB_DATA_DIR.
    id_mapping_path: str  # Relative to VDB_DATA_DIR.
    nprobe: int = 100
    normalize: bool = True
    enabled: bool = True


class VdbSettings(BaseModel):
    collections: dict[str, VdbCollectionSettings] = Field(default_factory=dict)


class ModelSettings(BaseModel):
    display_name: str
    hf_tokenizer_id: str
    max_length: int = 1022
    device_env_var: str
    checkpoint_env_var: str | None = None
    enabled: bool = True
    # Load precision for the backbone. Defaults to fp32; bf16 / fp16 halve
    # both weight and activation VRAM with negligible accuracy cost on the
    # PLM family we host. Per-model so ProtT5 (known fp16-stable) can be
    # bf16 while an experimental fine-tune stays fp32.
    precision: str = "fp32"
    # Optional precomputed per-residue embedding shard store. When set, the
    # embedding service can serve GET-by-UniRef50-id requests without
    # running the model. Intentionally orthogonal to `enabled`: a model can
    # have `enabled=false` and still serve its shards, so operators can
    # ship a cheap "embed-only-if-cached" deployment.
    shard_root: str | None = None
    shard_index: str | None = None  # defaults to `<shard_root>/index.db`
    shard_fallback_dirs: list[str] = []
    shard_dim: int | None = None  # defaults to PLM's own `dim` when omitted


class ModelRegistry(BaseModel):
    ankh_cl: ModelSettings
    ankh_large: ModelSettings
    esm1b: ModelSettings
    prott5: ModelSettings


class EmbeddingSettings(BaseModel):
    # Call torch.cuda.empty_cache() after each /embed response so the caching
    # allocator returns segments to the driver between requests. Small
    # per-request cost (~tens of ms to re-warm the pool) in exchange for
    # much lower resident VRAM when the embedding service shares a GPU
    # with other workloads. Defaults on since that's the case on deepfold.
    empty_cache_after_request: bool = True


class LimitsSettings(BaseModel):
    max_residues_per_chain: int = 1022
    max_chains_paired: int = 16
    max_body_bytes: int = 10 * 1024 * 1024


class CacheSettings(BaseModel):
    result_cache_max_size_gb: int = 100
    result_cache_max_age_days: int = 30
    embedding_cache_max_size_gb: int = 200
    embedding_cache_max_age_days: int = 90


class QueueSettings(BaseModel):
    worker_concurrency: int = 4
    backpressure_threshold: int = 50
    max_queue_depth: int = 200
    # Default FAISS neighbors fetched per model when the client omits `k` on
    # POST /v2/msa. Per-request `k` still wins and is capped at 10000 by the
    # API schema. 1000 matches upstream deepfold2's typical operating point.
    default_k: int = 1000
    # Target-embedding batch size. The worker splits each model's `k`
    # targets into chunks of this size before hitting /embed, so peak VRAM
    # is bounded by chunk_size * max_residues_per_chain * dim (not the full
    # k * max_residues * dim). Lower when GPU headroom is tight; raise when
    # round-trip overhead dominates.
    embed_chunk_size: int = 64
    # Within-job thread pool for the pairwise aligner. Each target's DP
    # runs independently; JIT-compiled kernels release the GIL so real
    # parallelism happens. 32 is a reasonable default on a modern
    # multi-core host: pLM-BLAST's per-target DP is ~1.5 s JIT'd, so
    # 500 targets fan out nicely. Set to 1 for repros; operators with
    # tight CPU budgets can cap lower.
    align_threads: int = 32
    # Paired-MSA retrieval multiplier. When the client submits
    # `paired=true`, each chain is retrieved at `paired_k = multiplier *
    # effective_k` (capped at the API-level max) so taxonomy filtering
    # still leaves a useful pool per chain. 3x matches MMseqs2
    # PairAlign's typical tuning; raise for sparser families, drop for
    # compute budget.
    paired_k_multiplier: int = 3


class RateLimitSettings(BaseModel):
    per_ip_rpm: int = 60
    per_token_rpm: int = 120


class CorsSettings(BaseModel):
    allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:*", "https://colab.research.google.com"]
    )
    allow_credentials: bool = False
    allow_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "DELETE", "OPTIONS"])
    allow_headers: list[str] = Field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-Request-ID"]
    )


class APISettings(BaseModel):
    v1_sunset_date: str = "2026-12-31"
    openapi_public: bool = True
    # Default token TTL (seconds) applied when a mint request omits
    # `expires_at`. 90d is a conservative default; operators can set to 0 to
    # opt out and mint non-expiring tokens (not recommended for public use).
    default_token_ttl_s: int = 60 * 60 * 24 * 90


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_format: bool = True
    request_id_header: str = "X-Request-ID"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=str(_resolve_settings_file()),
        env_nested_delimiter="__",
        extra="ignore",
    )

    service: ServiceSettings = ServiceSettings()
    models: ModelRegistry
    aligners: AlignerRegistry
    vdb: VdbSettings = VdbSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    limits: LimitsSettings = LimitsSettings()
    cache: CacheSettings = CacheSettings()
    queue: QueueSettings = QueueSettings()
    ratelimit: RateLimitSettings = RateLimitSettings()
    cors: CorsSettings = CorsSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
