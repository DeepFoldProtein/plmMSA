from __future__ import annotations

from fastapi.testclient import TestClient

from plmmsa.align.otalign import OTalign
from plmmsa.align.plmalign import PLMAlign
from plmmsa.align.server import create_app


def test_align_health_lists_aligners() -> None:
    app = create_app(aligners_override={"plmalign": PLMAlign(), "otalign": OTalign()})
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "align"
    assert "plmalign" in body["aligners"]
    assert "otalign" in body["aligners"]


def test_align_plmalign_roundtrip() -> None:
    app = create_app(aligners_override={"plmalign": PLMAlign()})
    payload = {
        "aligner": "plmalign",
        "mode": "local",
        # 3-residue query vs. 3-residue identical target; dim=2.
        "query_embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "target_embeddings": [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        "options": {"gap_open": 1.0, "gap_extend": 0.5},
    }
    with TestClient(app) as client:
        resp = client.post("/align", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["aligner"] == "plmalign"
    assert body["mode"] == "local"
    assert len(body["alignments"]) == 1
    a = body["alignments"][0]
    assert a["query_end"] == 3
    assert a["target_end"] == 3
    # All columns are match columns (no gaps).
    assert all(qi >= 0 and ti >= 0 for qi, ti in a["columns"])


def test_align_otalign_returns_501() -> None:
    app = create_app(aligners_override={"otalign": OTalign()})
    payload = {
        "aligner": "otalign",
        "mode": "local",
        "query_embedding": [[1.0, 0.0]],
        "target_embeddings": [[[1.0, 0.0]]],
    }
    with TestClient(app) as client:
        resp = client.post("/align", json=payload)
    assert resp.status_code == 501
    assert resp.json()["code"] == "E_NOT_IMPLEMENTED"


def test_align_unknown_aligner() -> None:
    app = create_app(aligners_override={"plmalign": PLMAlign()})
    with TestClient(app) as client:
        resp = client.post(
            "/align",
            json={
                "aligner": "bogus",
                "mode": "local",
                "query_embedding": [[1.0, 0.0]],
                "target_embeddings": [[[1.0, 0.0]]],
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_UNSUPPORTED_MODEL"
    assert body["detail"]["requested"] == "bogus"


def test_align_mismatched_dims() -> None:
    app = create_app(aligners_override={"plmalign": PLMAlign()})
    with TestClient(app) as client:
        resp = client.post(
            "/align",
            json={
                "aligner": "plmalign",
                "mode": "local",
                "query_embedding": [[1.0, 0.0]],
                "target_embeddings": [[[1.0, 0.0, 0.0]]],
            },
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "E_INVALID_FASTA"
