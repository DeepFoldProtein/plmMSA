"""Binary wire format for service-to-service tensor transport.

JSON transport of large per-residue embedding tensors (500-1500 of them,
each ~400 by 1024 floats) is ~1-3 GB and parses at ~50 MB/s in pure
Python — an order of magnitude more expensive than the actual compute
it surrounds. This module defines a compact framing used by
`/align/bin` (align service input) and `/embed_by_id/bin` (embedding
service output), with zero-copy encode/decode through
`numpy.ndarray.tobytes` / `numpy.frombuffer`.

Frame layout
------------
    magic         4 bytes   b"PLMA"
    version       u32       (currently 1)
    json_len      u32       metadata JSON length
    metadata      <json_len bytes>  UTF-8 JSON (endpoint-specific)
    n_tensors     u32       total count of tensors
    for each tensor:
        ndim      u32       always 2 for per-residue embeddings
        shape     ndim * u32
        payload   prod(shape) * 4 bytes  float32 little-endian

Endpoint conventions for `metadata` + tensor order:

- `/align/bin` requests:
    metadata = {aligner, mode, options}
    tensors[0]   = query_embedding  (Lq, D)
    tensors[1:]  = target_embeddings (Lt, D)

- `/embed_by_id/bin` responses:
    metadata = {model, dim, found_ids: [ids in tensor order], missing: [ids]}
    tensors  = one per found id, in `found_ids` order

Little-endian for portability. No compression — embeddings are dense
float32 and compression cost exceeds the I/O saved.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np

MAGIC = b"PLMA"
VERSION = 1
CONTENT_TYPE = "application/x-plmmsa-align"
CONTENT_TYPE_EMBED = "application/x-plmmsa-embed"


def encode(
    metadata: dict[str, Any],
    query: np.ndarray,
    targets: Sequence[np.ndarray],
) -> bytes:
    """Pack metadata + query + targets into one byte string.

    Each array is cast to float32 C-contiguous before the `tobytes()` —
    the wire format is strictly f32, so callers pay the cast here once
    rather than on the other side of the HTTP hop.
    """
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    tensors = [_coerce(query), *(_coerce(t) for t in targets)]

    parts: list[bytes] = [
        MAGIC,
        struct.pack("<III", VERSION, len(meta_bytes), len(tensors)),
        meta_bytes,
    ]
    for arr in tensors:
        parts.append(struct.pack("<I", arr.ndim))
        parts.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def decode(blob: bytes) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
    """Reverse of `encode`. Returns `(metadata, query, targets)`.

    Raises `ValueError` on magic / version mismatch so the endpoint can
    400 with a meaningful error instead of a silent `struct.error`.
    """
    if not blob.startswith(MAGIC):
        raise ValueError(f"bad magic; expected {MAGIC!r}")
    pos = len(MAGIC)
    version, meta_len, n_tensors = struct.unpack_from("<III", blob, pos)
    pos += 12
    if version != VERSION:
        raise ValueError(f"unsupported frame version {version}; expected {VERSION}")

    meta = json.loads(blob[pos:pos + meta_len].decode("utf-8"))
    pos += meta_len
    if n_tensors < 1:
        raise ValueError("frame has zero tensors; query is required")

    tensors: list[np.ndarray] = []
    for _ in range(n_tensors):
        (ndim,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        shape = struct.unpack_from(f"<{ndim}I", blob, pos)
        pos += 4 * ndim
        count = 1
        for d in shape:
            count *= d
        nbytes = count * 4
        # np.frombuffer(view) is zero-copy but the view is read-only;
        # copy into a writable array so downstream code (numba, any
        # in-place ops) doesn't trip.
        arr = np.frombuffer(blob, dtype=np.float32, count=count, offset=pos)
        tensors.append(arr.reshape(shape).copy())
        pos += nbytes

    if pos != len(blob):
        raise ValueError(f"trailing bytes in frame: {len(blob) - pos}")

    return meta, tensors[0], tensors[1:]


def _coerce(arr: Any) -> np.ndarray:
    """Cast + contiguous-ify so `tobytes` lays out the intended memory."""
    out = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return out


def encode_tensors(metadata: dict[str, Any], tensors: Sequence[np.ndarray]) -> bytes:
    """Generic variant of `encode` for endpoints that carry N tensors
    with no distinguished 'query' role. Used by `/embed_by_id/bin`
    responses where tensors are a flat list keyed by `metadata[found_ids]`.
    """
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    coerced = [_coerce(t) for t in tensors]
    parts: list[bytes] = [
        MAGIC,
        struct.pack("<III", VERSION, len(meta_bytes), len(coerced)),
        meta_bytes,
    ]
    for arr in coerced:
        parts.append(struct.pack("<I", arr.ndim))
        parts.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def decode_tensors(blob: bytes) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Reverse of `encode_tensors`. Returns `(metadata, tensors)`."""
    if not blob.startswith(MAGIC):
        raise ValueError(f"bad magic; expected {MAGIC!r}")
    pos = len(MAGIC)
    version, meta_len, n_tensors = struct.unpack_from("<III", blob, pos)
    pos += 12
    if version != VERSION:
        raise ValueError(f"unsupported frame version {version}; expected {VERSION}")

    meta = json.loads(blob[pos:pos + meta_len].decode("utf-8"))
    pos += meta_len

    tensors: list[np.ndarray] = []
    for _ in range(n_tensors):
        (ndim,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        shape = struct.unpack_from(f"<{ndim}I", blob, pos)
        pos += 4 * ndim
        count = 1
        for d in shape:
            count *= d
        nbytes = count * 4
        arr = np.frombuffer(blob, dtype=np.float32, count=count, offset=pos)
        tensors.append(arr.reshape(shape).copy())
        pos += nbytes

    if pos != len(blob):
        raise ValueError(f"trailing bytes in frame: {len(blob) - pos}")

    return meta, tensors


__all__ = [
    "CONTENT_TYPE",
    "CONTENT_TYPE_EMBED",
    "MAGIC",
    "VERSION",
    "decode",
    "decode_tensors",
    "encode",
    "encode_tensors",
]
