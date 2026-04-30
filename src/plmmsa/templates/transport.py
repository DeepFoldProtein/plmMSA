"""Transport layer for the templates re-alignment orchestrator.

`TemplatesTransport` is the narrow interface the orchestrator calls
into for the two upstream services it needs:

  - `embed(model, sequences)` → per-residue PLM embeddings
  - `align(aligner, mode, query, targets, options)` → pairwise alignments

`HttpTransport` is the production implementation — it uses the binary
wire formats (`/embed/bin`, `/align/bin`) so the templates pipeline
exercises the same bytes the rest of the stack does. Tests inject
their own object with the same shape (no inheritance required).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from plmmsa.align import binary as align_binary


@runtime_checkable
class TemplatesTransport(Protocol):
    """Narrow interface the orchestrator calls into.

    Marked `runtime_checkable` so test stubs that match the shape pass
    `isinstance(t, TemplatesTransport)` without having to subclass.
    """

    async def embed(
        self, *, model: str, sequences: Sequence[str]
    ) -> list[np.ndarray]: ...

    async def align(
        self,
        *,
        aligner: str,
        mode: str,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class HttpTransport:
    """Production transport — talks to the running services.

    Embed batching: sequences are length-sorted descending so each
    batch's padding cost is set by its longest member, then chunked at
    `embed_chunk_size` to bound peak GPU memory. Same pattern as
    `plmmsa.pipeline.orchestrator._embed_chunks`.

    Align: one call regardless of target count — the align service
    handles thread-pool fanout internally.
    """

    def __init__(
        self,
        *,
        embedding_url: str,
        align_url: str,
        timeout_s: float = 600.0,
        embed_chunk_size: int = 64,
    ) -> None:
        self._embedding_url = embedding_url
        self._align_url = align_url
        self._timeout_s = timeout_s
        self._embed_chunk_size = max(1, int(embed_chunk_size))

    async def embed(
        self, *, model: str, sequences: Sequence[str]
    ) -> list[np.ndarray]:
        if not sequences:
            return []
        # Length-sorted chunking: each batch's padding is its longest
        # member. Track original positions so callers see results in
        # the same order they sent.
        order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))
        sorted_seqs = [sequences[i] for i in order]
        sorted_out: list[np.ndarray] = []

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            for start in range(0, len(sorted_seqs), self._embed_chunk_size):
                batch = sorted_seqs[start : start + self._embed_chunk_size]
                resp = await client.post(
                    f"{self._embedding_url}/embed/bin",
                    json={"model": model, "sequences": batch},
                )
                resp.raise_for_status()
                _meta, tensors = align_binary.decode_tensors(resp.content)
                sorted_out.extend(np.asarray(t, dtype=np.float32) for t in tensors)

        # Restore caller-expected order.
        out: list[np.ndarray | None] = [None] * len(sequences)
        for pos, orig_i in enumerate(order):
            out[orig_i] = sorted_out[pos]
        return [t for t in out if t is not None]

    async def align(
        self,
        *,
        aligner: str,
        mode: str,
        query_embedding: np.ndarray,
        target_embeddings: Sequence[np.ndarray],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        body = align_binary.encode(
            {"aligner": aligner, "mode": mode, "options": dict(options)},
            query_embedding,
            list(target_embeddings),
        )
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._align_url}/align/bin",
                content=body,
                headers={"Content-Type": align_binary.CONTENT_TYPE},
            )
            resp.raise_for_status()
        return list(resp.json()["alignments"])


__all__ = ["HttpTransport", "TemplatesTransport"]
