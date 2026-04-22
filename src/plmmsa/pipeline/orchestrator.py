from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from plmmsa.jobs.models import JobResult
from plmmsa.pipeline.a3m import AlignmentHit, assemble_a3m
from plmmsa.pipeline.fetcher import TargetFetcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrchestratorConfig:
    embedding_url: str
    vdb_url: str
    align_url: str
    http_timeout: float = 120.0
    default_model: str = "ankh_cl"
    default_aligner: str = "plmalign"
    default_mode: str = "local"
    default_k: int = 100


class Orchestrator:
    """Wires embedding → vdb → (fetch) → embedding → align → A3M assembly."""

    def __init__(
        self,
        config: OrchestratorConfig,
        fetcher: TargetFetcher,
        *,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._config = config
        self._fetcher = fetcher
        self._client_factory = client_factory

    def _new_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(timeout=self._config.http_timeout)

    async def run(self, request: dict[str, Any]) -> JobResult:
        cfg = self._config
        sequences = request.get("sequences") or []
        if not sequences:
            raise ValueError("request.sequences is empty")

        # Only the first chain is pipelined today. Paired / per-chain handling
        # is a deferred PLAN.md item.
        query_seq = sequences[0]
        query_id = str(request.get("query_id") or "query")
        model = str(request.get("model") or cfg.default_model)
        aligner = str(request.get("aligner") or cfg.default_aligner)
        mode = str(request.get("mode") or cfg.default_mode)
        k = int(request.get("k") or cfg.default_k)
        collection = str(request.get("collection") or f"{model}_uniref50")

        async with self._new_client() as http:
            query_emb = await self._embed_query(http, model, query_seq)
            neighbors = await self._search(http, collection, query_emb, k)
            neighbor_ids = [n["id"] for n in neighbors]

            fetched = await self._fetcher.fetch(collection, neighbor_ids)
            kept_ids = [i for i in neighbor_ids if i in fetched]
            target_seqs = [fetched[i] for i in kept_ids]

            if not kept_ids:
                return _query_only_result(
                    query_id=query_id,
                    query_seq=query_seq,
                    query_self_score=float(len(query_seq)),
                    hits_found=len(neighbors),
                    hits_fetched=0,
                    model=model,
                    collection=collection,
                )

            target_embs = await self._embed_targets(http, model, target_seqs)
            alignments = await self._align(
                http, aligner, mode, query_emb, target_embs, request.get("options", {})
            )

        score_by_id = {n["id"]: float(n.get("distance", 0.0)) for n in neighbors}
        hits: list[AlignmentHit] = []
        for idx, tid in enumerate(kept_ids):
            a = alignments[idx]
            hits.append(
                AlignmentHit(
                    target_id=tid,
                    score=float(a.get("score", score_by_id.get(tid, 0.0))),
                    target_seq=target_seqs[idx],
                    columns=[(int(c[0]), int(c[1])) for c in a["columns"]],
                )
            )

        # Normalized per-residue embeddings have cosine self-similarity 1 per
        # diagonal cell, so the self-alignment score ~ len(query). Using the
        # exact trace via embeddings adds no information here.
        query_self_score = float(len(query_seq))
        a3m = assemble_a3m(
            query_id=query_id,
            query_seq=query_seq,
            query_self_score=query_self_score,
            hits=hits,
        )

        return JobResult(
            format="a3m",
            payload=a3m,
            stats={
                "pipeline": "orchestrator",
                "depth": 1 + len(hits),
                "hits_found": len(neighbors),
                "hits_fetched": len(hits),
                "model": model,
                "collection": collection,
                "aligner": aligner,
                "mode": mode,
            },
        )

    # --- Service calls ----------------------------------------------------

    async def _embed_query(self, http: httpx.AsyncClient, model: str, seq: str) -> np.ndarray:
        emb = await self._embed_targets(http, model, [seq])
        if emb[0].shape[0] != len(seq):
            raise ValueError(
                f"embedding length {emb[0].shape[0]} does not match query length {len(seq)}"
            )
        return emb[0]

    async def _embed_targets(
        self, http: httpx.AsyncClient, model: str, seqs: list[str]
    ) -> list[np.ndarray]:
        resp = await http.post(
            f"{self._config.embedding_url}/embed",
            json={"model": model, "sequences": seqs},
        )
        resp.raise_for_status()
        body = resp.json()
        return [np.asarray(e, dtype=np.float32) for e in body["embeddings"]]

    async def _search(
        self,
        http: httpx.AsyncClient,
        collection: str,
        query_emb: np.ndarray,
        k: int,
    ) -> list[dict[str, Any]]:
        pooled = query_emb.mean(axis=0, keepdims=True)
        resp = await http.post(
            f"{self._config.vdb_url}/search",
            json={"collection": collection, "vectors": pooled.tolist(), "k": k},
        )
        resp.raise_for_status()
        return resp.json()["results"][0]

    async def _align(
        self,
        http: httpx.AsyncClient,
        aligner: str,
        mode: str,
        query_emb: np.ndarray,
        target_embs: list[np.ndarray],
        options: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resp = await http.post(
            f"{self._config.align_url}/align",
            json={
                "aligner": aligner,
                "mode": mode,
                "query_embedding": query_emb.tolist(),
                "target_embeddings": [t.tolist() for t in target_embs],
                "options": dict(options),
            },
        )
        resp.raise_for_status()
        return resp.json()["alignments"]


def _query_only_result(
    *,
    query_id: str,
    query_seq: str,
    query_self_score: float,
    hits_found: int,
    hits_fetched: int,
    model: str,
    collection: str,
) -> JobResult:
    a3m = assemble_a3m(
        query_id=query_id,
        query_seq=query_seq,
        query_self_score=query_self_score,
        hits=[],
    )
    return JobResult(
        format="a3m",
        payload=a3m,
        stats={
            "pipeline": "orchestrator",
            "depth": 1,
            "hits_found": hits_found,
            "hits_fetched": hits_fetched,
            "model": model,
            "collection": collection,
            "note": "No target sequences could be fetched; MSA contains only the query.",
        },
    )
