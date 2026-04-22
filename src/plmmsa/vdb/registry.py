from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from plmmsa.config import Settings, VdbCollectionSettings
from plmmsa.vdb.base import VDB
from plmmsa.vdb.faiss_vdb import FaissVDB

logger = logging.getLogger(__name__)


def _resolve_path(data_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else data_dir / p


def _load_collection(
    collection_id: str,
    cfg: VdbCollectionSettings,
    data_dir: Path,
) -> VDB:
    return FaissVDB(
        collection_id=collection_id,
        display_name=cfg.display_name,
        model_backend=cfg.model_backend,
        dim=cfg.dim,
        index_path=_resolve_path(data_dir, cfg.index_path),
        id_mapping_path=_resolve_path(data_dir, cfg.id_mapping_path),
        default_nprobe=cfg.nprobe,
        normalize=cfg.normalize,
    )


def load_enabled_collections(
    settings: Settings,
    env: Mapping[str, str] | None = None,
) -> dict[str, VDB]:
    """Load every enabled collection from `settings.vdb.collections`.

    `VDB_DATA_DIR` (from env or defaulted to `./vdb_data`) is the base for
    collection index/mapping paths. Per-collection load failures are logged
    and skipped so the service starts with whatever loaded.
    """
    resolved_env = dict(env) if env is not None else dict(os.environ)
    data_dir = Path(resolved_env.get("VDB_DATA_DIR", "./vdb_data"))

    loaded: dict[str, VDB] = {}
    for name, cfg in settings.vdb.collections.items():
        if not cfg.enabled:
            logger.info("vdb.registry: %s is disabled in settings", name)
            continue
        try:
            loaded[name] = _load_collection(name, cfg, data_dir)
        except Exception:
            logger.exception("vdb.registry: failed to load %s", name)
    return loaded
