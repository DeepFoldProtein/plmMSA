"""Backwards-compat shim — metrics moved to ``plmmsa.metrics`` when they
were extended to cover every service. Keeps existing ``from
plmmsa.api.metrics import ...`` imports working.
"""

from plmmsa.metrics import (
    INFLIGHT,
    LATENCY,
    REGISTRY,
    REQUESTS,
    MetricsMiddleware,
    router,
)

__all__ = ["INFLIGHT", "LATENCY", "REGISTRY", "REQUESTS", "MetricsMiddleware", "router"]
