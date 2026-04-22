from __future__ import annotations

import uvicorn

from plmmsa.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "plmmsa.align.server:create_app",
        host="0.0.0.0",
        port=settings.service.align_port,
        log_level=settings.logging.level.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
