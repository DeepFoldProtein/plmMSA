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


class AlignerRegistry(BaseModel):
    plmalign: AlignerEntry
    otalign: AlignerEntry


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


class ModelRegistry(BaseModel):
    ankh_cl: ModelSettings
    ankh_large: ModelSettings
    esm1b: ModelSettings
    prott5: ModelSettings


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


class RateLimitSettings(BaseModel):
    per_ip_rpm: int = 30
    per_token_rpm: int = 120


class CorsSettings(BaseModel):
    allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:*", "https://colab.research.google.com"]
    )
    allow_credentials: bool = False


class APISettings(BaseModel):
    v1_sunset_date: str = "2026-12-31"
    openapi_public: bool = True


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
