from __future__ import annotations

import os

import uvicorn

from plmmsa.config import get_settings


def main() -> None:
    settings = get_settings()
    # uvloop is a drop-in faster event loop — strictly better for I/O-
    # heavy workloads like the align service's binary request handling.
    # `workers > 1` runs multiple ASGI processes behind the same port,
    # letting the align service handle concurrent /align/bin requests
    # from many worker containers at once. Each worker process forks
    # the aligner registry (numpy + numba caches); 4 is plenty on a
    # single host and is light on memory. Override via env.
    workers = int(os.environ.get("PLMMSA_ALIGN_UVICORN_WORKERS", "4"))
    uvicorn.run(
        "plmmsa.align.server:create_app",
        host="0.0.0.0",
        port=settings.service.align_port,
        log_level=settings.logging.level.lower(),
        factory=True,
        loop="uvloop",
        workers=workers,
    )


if __name__ == "__main__":
    main()
