"""Binary-wire-format coverage for the align service.

The framing in `plmmsa.align.binary` replaces JSON transport of big
embedding tensors. These tests pin: roundtrip integrity, graceful
failure on bad magic / version, metadata survival.
"""

from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align import binary as align_binary


def test_roundtrip_preserves_arrays_and_metadata() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(7, 16)).astype(np.float32)
    ts = [
        rng.normal(size=(11, 16)).astype(np.float32),
        rng.normal(size=(9, 16)).astype(np.float32),
        rng.normal(size=(5, 16)).astype(np.float32),
    ]
    meta = {
        "aligner": "plmalign",
        "mode": "local",
        "options": {"gap_open": 10.0, "gap_extend": 1.0},
    }
    blob = align_binary.encode(meta, q, ts)
    assert isinstance(blob, bytes)

    meta2, q2, ts2 = align_binary.decode(blob)
    assert meta == meta2
    assert np.array_equal(q, q2)
    assert len(ts2) == len(ts)
    for a, b in zip(ts, ts2, strict=True):
        assert np.array_equal(a, b)


def test_roundtrip_preserves_dtype_float32() -> None:
    """Caller-supplied arrays may be f64; encode coerces to f32 for the
    wire, and decode returns f32. Pin that."""
    q = np.arange(12, dtype=np.float64).reshape(3, 4)
    blob = align_binary.encode({"aligner": "plmalign", "mode": "local"}, q, [q])
    _, q2, ts2 = align_binary.decode(blob)
    assert q2.dtype == np.float32
    assert ts2[0].dtype == np.float32
    assert np.allclose(q2, q.astype(np.float32))


def test_decoded_arrays_are_writable() -> None:
    """np.frombuffer produces read-only views; decode must copy so the
    result works with in-place math and numba kernels."""
    q = np.ones((4, 8), dtype=np.float32)
    blob = align_binary.encode({}, q, [q])
    _, q2, _ = align_binary.decode(blob)
    # Writability check — will raise on read-only arrays.
    q2[0, 0] = 42.0
    assert q2[0, 0] == 42.0


def test_bad_magic_rejected() -> None:
    with pytest.raises(ValueError, match="bad magic"):
        align_binary.decode(b"XXXX" + b"\x00" * 128)


def test_unsupported_version_rejected() -> None:
    # Magic OK, version != 1.
    import struct

    bad = align_binary.MAGIC + struct.pack("<III", 99, 0, 1) + b""
    # Add a zero-dim tensor placeholder to reach the decode point.
    bad += struct.pack("<I", 0)
    with pytest.raises(ValueError, match="version"):
        align_binary.decode(bad)


def test_trailing_bytes_rejected() -> None:
    q = np.zeros((2, 2), dtype=np.float32)
    blob = align_binary.encode({}, q, [q]) + b"garbage"
    with pytest.raises(ValueError, match="trailing bytes"):
        align_binary.decode(blob)


def test_payload_size_cut_on_large_input() -> None:
    """Sanity check: the binary frame is meaningfully smaller than the
    equivalent JSON. Concrete numbers for a toy 10x64 x 100 payload."""
    import json as _json

    q = np.random.default_rng(0).normal(size=(10, 64)).astype(np.float32)
    ts = [q.copy() for _ in range(100)]
    binary_size = len(align_binary.encode({"aligner": "plmalign"}, q, ts))
    json_size = len(
        _json.dumps(
            {
                "query": q.tolist(),
                "targets": [t.tolist() for t in ts],
            }
        )
    )
    # JSON is typically 5-15x larger on f32 payloads (each float → ~12
    # chars with a comma). We only assert it's at least 3x as a lower
    # bound so the test isn't fragile across numpy / Python versions.
    assert json_size >= 3 * binary_size
