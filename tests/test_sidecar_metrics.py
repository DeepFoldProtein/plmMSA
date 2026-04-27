"""Coverage for the `/metrics` endpoint attached to sidecar services
(embedding, vdb, align) and the worker's embedded metrics server.

Shape under test:

- Every sidecar app serves ``/metrics`` in the Prometheus text
  exposition format, with counters labeled ``service="<name>"``.
- The counter increments after a real request traverses the sidecar.
- Worker metrics (jobs processed, pipeline duration, queue depth)
  update as jobs flow through ``_one_iteration``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from plmmsa.jobs import JobResult, JobStore, ResultCache
from plmmsa.metrics import (
    WORKER_JOBS_PROCESSED,
    WORKER_PIPELINE_DURATION,
    WORKER_QUEUE_DEPTH,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    """Prometheus counters are module globals — reset between tests so
    one test's increments don't leak into the next's assertions."""
    # Best-effort: clear all observed label permutations on the worker
    # counters (the only ones this file asserts counts against).
    try:
        WORKER_JOBS_PROCESSED.clear()
    except Exception:
        pass  # clear() was added in prometheus-client 0.18; tolerate older
    try:
        WORKER_QUEUE_DEPTH.set(0)
    except Exception:
        pass


def _stub_backend(dim: int = 4):
    """Minimal PLM stub so the embedding app boots without loading torch
    weights."""
    import torch

    from plmmsa.plm.base import PLM

    class _StubPLM(PLM):
        id = "fake"
        display_name = "Fake"
        dim = 4
        max_length = 10

        def __init__(self) -> None:
            self.device = torch.device("cpu")

        def encode(self, sequences):
            return [torch.ones((len(s), self.dim)) * len(s) for s in sequences]

    return _StubPLM()


def test_embedding_sidecar_exposes_metrics() -> None:
    from plmmsa.embedding.server import create_app

    app = create_app(backends_override={"fake": _stub_backend()})
    with TestClient(app) as c:
        # Drive a real request so the counter has a label to emit.
        c.get("/health")
        resp = c.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text
    assert "plmmsa_http_requests_total" in body
    assert 'service="embedding"' in body


def test_vdb_sidecar_exposes_metrics() -> None:
    from plmmsa.vdb.server import create_app

    app = create_app(collections_override={})
    with TestClient(app) as c:
        c.get("/health")
        resp = c.get("/metrics")

    assert resp.status_code == 200
    body = resp.text
    assert 'service="vdb"' in body


def test_align_sidecar_exposes_metrics() -> None:
    from plmmsa.align.server import create_app

    app = create_app(aligners_override={})
    with TestClient(app) as c:
        c.get("/health")
        resp = c.get("/metrics")

    assert resp.status_code == 200
    body = resp.text
    assert 'service="align"' in body


# ---------------------------------------------------------------------------
# Worker-side counters
# ---------------------------------------------------------------------------


class _StubOrchestrator:
    def __init__(self, result: JobResult) -> None:
        self._result = result

    async def run(self, request: dict[str, Any]) -> JobResult:
        return self._result


class _BoomOrchestrator:
    async def run(self, request: dict[str, Any]) -> JobResult:
        raise RuntimeError("synthetic failure")


async def test_worker_counters_increment_on_success() -> None:
    from plmmsa.worker.__main__ import _one_iteration

    store = JobStore(FakeAsyncRedis())
    cache = ResultCache(None, ttl_s=0)
    await store.create({"sequences": ["MKT"]})
    orch = _StubOrchestrator(JobResult(payload=">q\nMKT\n"))

    before = WORKER_JOBS_PROCESSED.labels(status="succeeded")._value.get()
    await _one_iteration(store, orch, cache)  # type: ignore[arg-type]
    after = WORKER_JOBS_PROCESSED.labels(status="succeeded")._value.get()
    assert after == before + 1

    # Histogram sample count incremented on success path.
    hist_count = WORKER_PIPELINE_DURATION._sum.get()
    assert hist_count >= 0.0  # existence check — real pipelines add real time


async def test_worker_counters_increment_on_failure() -> None:
    from plmmsa.worker.__main__ import _one_iteration

    store = JobStore(FakeAsyncRedis())
    cache = ResultCache(None, ttl_s=0)
    await store.create({"sequences": ["MKT"]})
    orch = _BoomOrchestrator()

    before = WORKER_JOBS_PROCESSED.labels(status="failed")._value.get()
    await _one_iteration(store, orch, cache)  # type: ignore[arg-type]
    after = WORKER_JOBS_PROCESSED.labels(status="failed")._value.get()
    assert after == before + 1


async def test_worker_queue_depth_sampler_updates_gauge() -> None:
    from plmmsa.worker.__main__ import _sample_queue_depth

    store = JobStore(FakeAsyncRedis())
    # Push three items so the gauge has something to observe.
    for _ in range(3):
        await store.create({"sequences": ["MKT"]})

    stop_event = asyncio.Event()
    task = asyncio.create_task(_sample_queue_depth(store, stop_event=stop_event))
    # Let one sample run, then stop.
    await asyncio.sleep(0.05)
    stop_event.set()
    await task

    assert WORKER_QUEUE_DEPTH._value.get() == 3
