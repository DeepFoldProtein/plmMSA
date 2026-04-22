from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping

from plmmsa.config import ModelSettings, Settings
from plmmsa.plm.ankh_cl import AnkhCL
from plmmsa.plm.ankh_large import AnkhLarge
from plmmsa.plm.base import PLM
from plmmsa.plm.esm1b import ESM1b
from plmmsa.plm.prott5 import ProtT5

logger = logging.getLogger(__name__)


def _load_ankh_cl(cfg: ModelSettings, env: Mapping[str, str]) -> PLM:
    checkpoint = env.get(cfg.checkpoint_env_var or "", "")
    device = env.get(cfg.device_env_var, "cpu")
    return AnkhCL(checkpoint=checkpoint, device=device)


def _load_ankh_large(cfg: ModelSettings, env: Mapping[str, str]) -> PLM:
    return AnkhLarge(device=env.get(cfg.device_env_var, "cpu"))


def _load_esm1b(cfg: ModelSettings, env: Mapping[str, str]) -> PLM:
    return ESM1b(device=env.get(cfg.device_env_var, "cpu"))


def _load_prott5(cfg: ModelSettings, env: Mapping[str, str]) -> PLM:
    return ProtT5(device=env.get(cfg.device_env_var, "cpu"))


LOADERS: dict[str, Callable[[ModelSettings, Mapping[str, str]], PLM]] = {
    "ankh_cl": _load_ankh_cl,
    "ankh_large": _load_ankh_large,
    "esm1b": _load_esm1b,
    "prott5": _load_prott5,
}


def load_enabled_backends(
    settings: Settings,
    env: Mapping[str, str] | None = None,
) -> dict[str, PLM]:
    """Instantiate every enabled PLM backend declared in `settings.models`.

    Load failures are logged and skipped — the embedding server starts with
    whatever loaded. Callers decide what to do about a partial load via
    the `/health` aggregate.
    """
    resolved_env = dict(env) if env is not None else dict(os.environ)
    loaded: dict[str, PLM] = {}
    for name, loader in LOADERS.items():
        cfg: ModelSettings = getattr(settings.models, name)
        if not cfg.enabled:
            logger.info("plm.registry: %s is disabled in settings", name)
            continue
        try:
            loaded[name] = loader(cfg, resolved_env)
        except Exception:
            logger.exception("plm.registry: failed to load %s", name)
    return loaded
