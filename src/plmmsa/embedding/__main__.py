from __future__ import annotations

import uvicorn

from plmmsa.config import get_settings


def main() -> None:
    settings = get_settings()
    # uvloop for the event loop. Workers=1 intentionally — each worker
    # process would load a full set of PLMs (~30 GB on GPU), so process-
    # fanning is not an option until we're ready to split models across
    # workers. asyncio.gather inside the handler is doing the real
    # parallelism for shard disk reads.
    uvicorn.run(
        "plmmsa.embedding.server:create_app",
        host="0.0.0.0",
        port=settings.service.embedding_port,
        log_level=settings.logging.level.lower(),
        factory=True,
        loop="uvloop",
    )


if __name__ == "__main__":
    main()
