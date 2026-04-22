from __future__ import annotations

import uvicorn

from plmmsa.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "plmmsa.embedding.server:create_app",
        host="0.0.0.0",
        port=settings.service.embedding_port,
        log_level=settings.logging.level.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
