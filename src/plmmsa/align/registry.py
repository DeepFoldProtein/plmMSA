from __future__ import annotations

import logging
from collections.abc import Callable

from plmmsa.align.base import Aligner
from plmmsa.align.otalign import OTalign
from plmmsa.align.plmalign import PLMAlign
from plmmsa.config import AlignerEntry, Settings

logger = logging.getLogger(__name__)


LOADERS: dict[str, Callable[[AlignerEntry], Aligner]] = {
    "plmalign": lambda _cfg: PLMAlign(),
    "otalign": lambda _cfg: OTalign(),
}


def load_enabled_aligners(settings: Settings) -> dict[str, Aligner]:
    """Instantiate every enabled aligner declared in `settings.aligners`.

    Load failures are logged and skipped — the align service starts with
    whatever loaded. `/health` reports the final set.
    """
    loaded: dict[str, Aligner] = {}
    for name, loader in LOADERS.items():
        cfg: AlignerEntry = getattr(settings.aligners, name)
        if not cfg.enabled:
            logger.info("align.registry: %s is disabled in settings", name)
            continue
        try:
            loaded[name] = loader(cfg)
        except Exception:
            logger.exception("align.registry: failed to load %s", name)
    return loaded
