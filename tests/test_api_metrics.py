"""Prometheus /metrics endpoint smoke coverage.

Verifies the endpoint serves text exposition format, that the request
counter increments after a round-trip through the stack, and that it is
exempt from auth + rate-limit middleware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from plmmsa.api import app

    return TestClient(app)


def test_metrics_endpoint_is_anonymous(client: TestClient) -> None:
    with client as c:
        resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text
    assert "plmmsa_http_requests_total" in body


def test_metrics_counter_increments(client: TestClient) -> None:
    with client as c:
        c.get("/v2/version")  # drive one request through
        resp = c.get("/metrics")
    body = resp.text
    # The counter line we want looks like:
    #   plmmsa_http_requests_total{method="GET",route="/v2/version",service="api",status="200"} 1.0
    assert 'route="/v2/version"' in body
    assert 'service="api"' in body
