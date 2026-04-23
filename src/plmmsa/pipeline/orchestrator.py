from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from plmmsa.jobs.models import JobResult
from plmmsa.pipeline.a3m import AlignmentHit, assemble_a3m, assemble_paired_a3m
from plmmsa.pipeline.fetcher import TargetFetcher
from plmmsa.pipeline.paired import join_by_taxonomy

logger = logging.getLogger(__name__)


DEFAULT_ALIGNER_ID = "plmalign"
# NOTE: upstream PLMAlign's main pathfinder is our `plm_blast` (multi-path
# SW), not this `plmalign` (affine-gap SW). We keep `plmalign` as the
# default only for performance: pLM-BLAST's column/row-max gap term is
# O(Lq · Lt · max(Lq, Lt)) in pure Python — ~hours for k=500 targets
# without numba / torch acceleration (tracked in PLAN.md). Flip this
# constant once the DP is jit-accelerated or vectorized.


def default_aligner_id() -> str:
    """The orchestrator's default aligner id, safe for callers that don't
    want to instantiate an `OrchestratorConfig` just to read one field."""
    return DEFAULT_ALIGNER_ID


def _score_threshold(query_len: int) -> float:
    """Upstream PLMAlign's Algorithm 1 step 5 threshold:
    `min(0.2 * Sself, 8.0)` where Sself = len(Q). For any query
    >= 40 aa this saturates at 8.0.
    """
    return min(0.2 * float(query_len), 8.0)


def _columns_in_bounds(
    columns: list[tuple[int, int]],
    query_len: int,
    target_len: int,
) -> bool:
    """True when every column's (qi, ti) is within its sequence.

    Drops align responses that disagree with the cached target sequence's
    length — typically a symptom of `cache-seq` + the PLM shard store
    being populated from different UniRef50 snapshots. Rendering an
    out-of-bounds hit crashes `render_hit`; a defensive filter here
    lets the rest of the job proceed instead of taking the whole MSA
    down.
    """
    return all(qi < query_len and ti < target_len for qi, ti in columns)


@dataclass(slots=True)
class OrchestratorConfig:
    embedding_url: str
    vdb_url: str
    align_url: str
    # Covers both /embed (large-batch targets) and /align (OTalign on
    # long queries x ~1500 targets). OTalign on L=235 against 1500
    # targets can take 5-8 min; 120s was failing on CASP15 T1120.
    http_timeout: float = 900.0
    default_model: str = "ankh_cl"
    # pLM-BLAST is the upstream PLMAlign default (multi-path SW over a
    # Z-scored similarity matrix). Our affine-gap PLMAlign is still
    # registered; clients can opt back in via `aligner = "plmalign"`.
    default_aligner: str = DEFAULT_ALIGNER_ID
    default_mode: str = "local"
    default_k: int = 100
    # Cap on how many target sequences are batched per embedding request.
    # Higher = fewer round-trips, more peak GPU memory. 64 is a safe default
    # for ankh_cl on a 48 GB GPU when sequences are <= ~1000 residues.
    embed_chunk_size: int = 64
    # Models whose target embeddings should come from /embed_by_id first
    # (the precomputed shard store) before falling back to /embed. Opt-in
    # by env var so deployments without the shard mount keep today's
    # behavior unchanged.
    shard_models: frozenset[str] = frozenset()
    # Cross-PLM scoring: the PLM used to build the score matrix when
    # retrieval and scoring use different models. Empty = score with
    # whichever PLM searched (per-model aggregate path). Non-empty =
    # "upstream topology": parallel retrieval via every model in
    # `request.models`, union hits, then one alignment pass with this
    # model's embeddings.
    score_model: str = ""
    # Transport for /align request body. "binary" uses the compact
    # framing in `plmmsa.align.binary` (zero-copy decode, ~50x faster
    # than JSON on large payloads) and hits `/align/bin`. "json" sends
    # the legacy dict-of-lists body to `/align` — kept for tests and
    # interop with older align images.
    align_transport: str = "binary"
    # Per-aligner score-threshold filter (Algorithm 1 step 5). Holds
    # the aligner ids whose `[aligners.*].filter_enabled = true` in
    # settings.toml. Empty default keeps tests / non-API callers in
    # their pre-settings "no filter" behavior.
    filter_enabled_aligners: frozenset[str] = field(default_factory=frozenset)
    # Paired-MSA retrieval multiplier. Per-chain retrieval uses
    # `paired_k = multiplier * effective_k` so taxonomy-join filtering
    # still yields a useful pool. 3x matches upstream MMseqs2 tuning.
    paired_k_multiplier: int = 3
    # Hard cap matching the API schema's `k` limit; paired retrieval
    # respects this so operators stay in control of GPU load.
    paired_k_cap: int = 10_000


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

        # Per-chain labels — the API stamps these; old-shape callers
        # (bench scripts, tests) may still send `query_id` singular.
        # Default to incrementing integer labels matching upstream's
        # ColabFold convention ("101", "102", ...).
        query_ids_raw = request.get("query_ids")
        if query_ids_raw:
            query_ids = [str(q) for q in query_ids_raw]
        else:
            singular = request.get("query_id")
            if singular and len(sequences) == 1:
                query_ids = [str(singular)]
            else:
                query_ids = [str(101 + i) for i in range(len(sequences))]

        query_seq = sequences[0]
        query_id = query_ids[0]
        aligner = str(request.get("aligner") or cfg.default_aligner)
        mode = str(request.get("mode") or cfg.default_mode)
        k = int(request.get("k") or cfg.default_k)
        options = request.get("options") or {}
        # Upstream PLMAlign's score-threshold filter (Algorithm 1 step 5).
        # Default on; callers can opt out with filter_by_score=False.
        filter_by_score_raw = request.get("filter_by_score", True)
        filter_by_score = bool(filter_by_score_raw) if filter_by_score_raw is not None else True
        paired = bool(request.get("paired", False))

        # Resolve the effective model list. API edge normalizes this into
        # `request["models"]`; we keep the legacy `model` field as a fallback
        # so old-shape payloads (single PLM) still work.
        models = list(request.get("models") or [])
        if not models:
            models = [str(request.get("model") or cfg.default_model)]
        # Allow callers to pin collections per-model; otherwise fall back to
        # `<model>_uniref50`.
        collections_override = request.get("collections") or {}
        if not isinstance(collections_override, dict):
            collections_override = {}
        # Single-model callers can still pass `collection=` for back-compat.
        single_collection = request.get("collection")

        async def _resolve_collection(m: str) -> str:
            if single_collection and len(models) == 1:
                return str(single_collection)
            if m in collections_override:
                return str(collections_override[m])
            return f"{m}_uniref50"

        # Cross-PLM scoring: request override wins, else server default.
        # `None` request value means "use server default"; empty string
        # explicitly opts out.
        if "score_model" in request and request["score_model"] is not None:
            score_model = str(request["score_model"])
        else:
            score_model = cfg.score_model

        if paired and len(sequences) > 1:
            # Paired requires cross-PLM scoring — without it the
            # alignment step and taxonomy-join contract get fuzzy
            # (per-chain hits need comparable scores). Fall back to
            # the aligner's default score_model when caller didn't
            # specify one.
            if not score_model:
                # Paired needs comparable per-chain scores. Default to
                # ProtT5 since its shard store makes per-chain retrieval
                # cheap; operators can still pin via settings.
                score_model = cfg.score_model or "prott5"
            return await self._run_paired(
                query_seqs=[str(s) for s in sequences],
                query_ids=query_ids,
                retrieval_models=models,
                resolve_collection=_resolve_collection,
                score_model=score_model,
                k=k,
                aligner=aligner,
                mode=mode,
                options=options,
                filter_by_score=filter_by_score,
            )

        if score_model:
            return await self._run_cross_plm(
                query_seq=query_seq,
                query_id=query_id,
                retrieval_models=models,
                resolve_collection=_resolve_collection,
                score_model=score_model,
                k=k,
                aligner=aligner,
                mode=mode,
                options=options,
                filter_by_score=filter_by_score,
            )

        per_model: list[_PerModelResult] = []
        async with self._new_client() as http:
            tasks = [
                self._run_one_model(
                    http,
                    model=m,
                    collection=await _resolve_collection(m),
                    query_seq=query_seq,
                    k=k,
                    aligner=aligner,
                    mode=mode,
                    options=options,
                )
                for m in models
            ]
            per_model = list(await asyncio.gather(*tasks))

        # Union hits across models by target_id, keeping the highest-scoring
        # variant. We don't remove the losers — we just drop the duplicates,
        # so a sequence that came in via ankh_cl *and* esm1b shows up once
        # with the better alignment of the two.
        merged: dict[str, AlignmentHit] = {}
        for r in per_model:
            for hit in r.hits:
                existing = merged.get(hit.target_id)
                if existing is None or hit.score > existing.score:
                    merged[hit.target_id] = hit

        # Rank by score descending so the A3M reads "most confident first".
        merged_hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)

        if not merged_hits:
            return _query_only_result(
                query_id=query_id,
                query_seq=query_seq,
                query_self_score=float(len(query_seq)),
                hits_found=sum(r.hits_found for r in per_model),
                hits_fetched=0,
                models=models,
                per_model_stats={r.model: r.stats() for r in per_model},
            )

        # Upstream PLMAlign Algorithm 1 step 5. Per-aligner toggle via
        # `[aligners.*].filter_enabled` (OTalign off by default because
        # its score scale differs). Per-request `filter_by_score` must
        # also be true; either false disables the filter.
        hits_pre_filter = len(merged_hits)
        filter_threshold = _score_threshold(len(query_seq))
        filter_applied = filter_by_score and aligner in cfg.filter_enabled_aligners
        if filter_applied:
            merged_hits = [h for h in merged_hits if h.score >= filter_threshold]

        query_self_score = float(len(query_seq))
        a3m = assemble_a3m(
            query_id=query_id,
            query_seq=query_seq,
            query_self_score=query_self_score,
            hits=merged_hits,
        )

        return JobResult(
            format="a3m",
            payload=a3m,
            stats={
                "pipeline": "orchestrator",
                "depth": 1 + len(merged_hits),
                "hits_found": sum(r.hits_found for r in per_model),
                "hits_fetched": len(merged_hits),
                "hits_fetched_before_union": sum(len(r.hits) for r in per_model),
                "hits_pre_filter": hits_pre_filter,
                "hits_post_filter": len(merged_hits),
                "filter_by_score": filter_by_score,
                "filter_applied": filter_applied,
                "filter_threshold": filter_threshold,
                "models": models,
                "aligner": aligner,
                "mode": mode,
                "per_model": {r.model: r.stats() for r in per_model},
            },
        )

    async def _run_one_model(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        collection: str,
        query_seq: str,
        k: int,
        aligner: str,
        mode: str,
        options: dict[str, Any],
    ) -> _PerModelResult:
        """Run embed → search → fetch → embed → align for a single PLM.

        The merging step in `run` handles combining hits across models; this
        returns per-model `AlignmentHit`s plus the diagnostics the stats
        block surfaces."""
        try:
            query_emb = await self._embed_query(http, model, query_seq)
            neighbors = await self._search(http, collection, query_emb, k)
            neighbor_ids = [n["id"] for n in neighbors]

            fetched = await self._fetcher.fetch(collection, neighbor_ids)
            kept_ids = [i for i in neighbor_ids if i in fetched]
            target_seqs = [fetched[i] for i in kept_ids]

            if not kept_ids:
                return _PerModelResult(
                    model=model,
                    collection=collection,
                    hits_found=len(neighbors),
                    hits=[],
                )

            target_embs = await self._embed_targets(
                http,
                model,
                target_seqs,
                ids=kept_ids,
            )
            alignments = await self._align(http, aligner, mode, query_emb, target_embs, options)
        except Exception as exc:
            # A single model's failure should not nuke the whole request when
            # we're running multiple — record the error and let the union
            # return whatever other models produced.
            logger.warning("model %s pipeline failed: %s", model, exc)
            return _PerModelResult(
                model=model,
                collection=collection,
                hits_found=0,
                hits=[],
                error=str(exc)[:200],
            )

        score_by_id = {n["id"]: float(n.get("distance", 0.0)) for n in neighbors}
        hits: list[AlignmentHit] = []
        dropped_oob = 0
        for idx, tid in enumerate(kept_ids):
            a = alignments[idx]
            target_seq = target_seqs[idx]
            columns = [(int(c[0]), int(c[1])) for c in a["columns"]]
            if not _columns_in_bounds(columns, len(query_seq), len(target_seq)):
                dropped_oob += 1
                continue
            hits.append(
                AlignmentHit(
                    target_id=tid,
                    score=float(a.get("score", score_by_id.get(tid, 0.0))),
                    target_seq=target_seq,
                    columns=columns,
                )
            )
        if dropped_oob:
            logger.warning(
                "orchestrator: dropped %d/%d hits for model=%s with "
                "columns out of target-seq bounds (cache-seq / shard snapshot drift)",
                dropped_oob,
                len(kept_ids),
                model,
            )
        return _PerModelResult(
            model=model,
            collection=collection,
            hits_found=len(neighbors),
            hits=hits,
        )

    async def _run_cross_plm(
        self,
        *,
        query_seq: str,
        query_id: str,
        retrieval_models: list[str],
        resolve_collection: Any,  # async (str) -> str
        score_model: str,
        k: int,
        aligner: str,
        mode: str,
        options: dict[str, Any],
        filter_by_score: bool = True,
    ) -> JobResult:
        """Cross-PLM topology wrapper — computes per-query hits via
        `_cross_plm_hits` then assembles the single-chain A3M. See that
        method's docstring for the pipeline shape.
        """
        result = await self._cross_plm_hits(
            query_seq=query_seq,
            retrieval_models=retrieval_models,
            resolve_collection=resolve_collection,
            score_model=score_model,
            k=k,
            aligner=aligner,
            mode=mode,
            options=options,
            filter_by_score=filter_by_score,
        )
        if not result.hits and not result.hits_fetched:
            return _query_only_result(
                query_id=query_id,
                query_seq=query_seq,
                query_self_score=float(len(query_seq)),
                hits_found=result.hits_found,
                hits_fetched=0,
                models=retrieval_models,
                per_model_stats=result.per_retrieval_stats,
            )
        a3m = assemble_a3m(
            query_id=query_id,
            query_seq=query_seq,
            query_self_score=float(len(query_seq)),
            hits=result.hits,
        )
        return JobResult(
            format="a3m",
            payload=a3m,
            stats={
                "pipeline": "orchestrator",
                "topology": "cross_plm",
                "depth": 1 + len(result.hits),
                "hits_found": result.hits_found,
                "hits_fetched": len(result.hits),
                "hits_pre_filter": result.hits_pre_filter,
                "hits_post_filter": len(result.hits),
                "filter_by_score": filter_by_score,
                "filter_applied": result.filter_applied,
                "filter_threshold": result.filter_threshold,
                "retrieval_models": retrieval_models,
                "score_model": score_model,
                "aligner": aligner,
                "mode": mode,
                "per_retrieval": result.per_retrieval_stats,
            },
        )

    async def _cross_plm_hits(
        self,
        *,
        query_seq: str,
        retrieval_models: list[str],
        resolve_collection: Any,  # async (str) -> str
        score_model: str,
        k: int,
        aligner: str,
        mode: str,
        options: dict[str, Any],
        filter_by_score: bool,
    ) -> _CrossPlmChainResult:
        """Upstream PLMAlign topology — returns the post-filter hit list.

        Parallel VDB searches across `retrieval_models` → union hit ids →
        single re-embed pass with `score_model` → one alignment call.
        No A3M assembly; callers (single-chain and paired) wrap the
        returned hit list in their own renderer.

        Benefits vs. per-model-aggregate:
          - One alignment pass instead of N — cheaper and scores are
            directly comparable across hits.
          - Shard-store shortcut is maximally useful: target embeddings
            almost always come from disk when `score_model` has a shard
            mount configured.

        Cost:
          - One extra /embed call per retrieval model (for the query
            residues). Amortizes over k hits so negligible at k>=500.
        """
        async with self._new_client() as http:
            # Phase 1 — parallel retrieval across VDBs. In parallel we
            # also kick off the score_model query embed (it doesn't
            # depend on retrieval results), so it's ready by the time
            # we need it in Phase 4. Saves one full embed round-trip
            # on the critical path for cold jobs.
            async def _retrieve(
                rm: str,
            ) -> tuple[str, str, list[dict[str, Any]] | None, str | None]:
                collection = await resolve_collection(rm)
                try:
                    q_emb = await self._embed_query(http, rm, query_seq)
                    neighbors = await self._search(http, collection, q_emb, k)
                    return rm, collection, neighbors, None
                except Exception as exc:
                    logger.warning("retrieval model %s failed: %s", rm, exc)
                    return rm, collection, None, str(exc)[:200]

            score_query_task = asyncio.create_task(self._embed_query(http, score_model, query_seq))
            retrievals = await asyncio.gather(*(_retrieve(rm) for rm in retrieval_models))

            # Phase 2 — union ids across VDBs. Order preserved by first
            # occurrence, keep best-distance-per-id for reference stats.
            best_distance: dict[str, float] = {}
            order: list[str] = []
            per_retrieval_stats: dict[str, dict[str, Any]] = {}
            for rm, collection, neighbors, err in retrievals:
                stats_entry: dict[str, Any] = {
                    "collection": collection,
                    "hits_found": 0 if neighbors is None else len(neighbors),
                }
                if err is not None:
                    stats_entry["error"] = err
                per_retrieval_stats[rm] = stats_entry
                if neighbors is None:
                    continue
                for n in neighbors:
                    tid = n["id"]
                    dist = float(n.get("distance", 0.0))
                    if tid not in best_distance:
                        best_distance[tid] = dist
                        order.append(tid)
                    elif dist < best_distance[tid]:
                        best_distance[tid] = dist

            if not order:
                score_query_task.cancel()
                with contextlib.suppress(BaseException):
                    await score_query_task
                return _CrossPlmChainResult(
                    hits=[],
                    hits_found=0,
                    hits_fetched=0,
                    hits_pre_filter=0,
                    filter_applied=False,
                    filter_threshold=_score_threshold(len(query_seq)),
                    per_retrieval_stats=per_retrieval_stats,
                )

            # Phase 3 — fetch target sequences. Fetcher key format is
            # collection-agnostic on our deployment (`seq:UniRef50_{id}`);
            # pass the first retrieval collection for anyone who keys by it.
            first_collection = next(
                iter(s["collection"] for s in per_retrieval_stats.values() if "collection" in s),
                f"{retrieval_models[0]}_uniref50",
            )
            fetched = await self._fetcher.fetch(first_collection, order)
            kept_ids = [tid for tid in order if tid in fetched]
            target_seqs = [fetched[tid] for tid in kept_ids]

            if not kept_ids:
                score_query_task.cancel()
                with contextlib.suppress(BaseException):
                    await score_query_task
                return _CrossPlmChainResult(
                    hits=[],
                    hits_found=len(order),
                    hits_fetched=0,
                    hits_pre_filter=0,
                    filter_applied=False,
                    filter_threshold=_score_threshold(len(query_seq)),
                    per_retrieval_stats=per_retrieval_stats,
                )

            # Phase 4 — score-model embed pass. Query embed was kicked
            # off in Phase 1 (runs concurrently with retrieval); just
            # await it now. Target embeds prefer the shard store.
            score_query_emb = await score_query_task
            score_target_embs = await self._embed_targets(
                http,
                score_model,
                target_seqs,
                ids=kept_ids,
            )

            # Phase 5 — one alignment call.
            alignments = await self._align(
                http, aligner, mode, score_query_emb, score_target_embs, options
            )

        hits: list[AlignmentHit] = []
        dropped_oob = 0
        for idx, tid in enumerate(kept_ids):
            a = alignments[idx]
            target_seq = target_seqs[idx]
            columns = [(int(c[0]), int(c[1])) for c in a["columns"]]
            if not _columns_in_bounds(columns, len(query_seq), len(target_seq)):
                dropped_oob += 1
                continue
            hits.append(
                AlignmentHit(
                    target_id=tid,
                    score=float(a.get("score", best_distance.get(tid, 0.0))),
                    target_seq=target_seq,
                    columns=columns,
                )
            )
        if dropped_oob:
            logger.warning(
                "orchestrator: cross-plm dropped %d/%d hits with "
                "columns out of target-seq bounds (cache-seq / shard snapshot drift)",
                dropped_oob,
                len(kept_ids),
            )
        hits.sort(key=lambda h: h.score, reverse=True)

        # Upstream PLMAlign Algorithm 1 step 5. Per-aligner toggle
        # via settings; per-request `filter_by_score` must also be true.
        hits_pre_filter = len(hits)
        filter_threshold = _score_threshold(len(query_seq))
        filter_applied = filter_by_score and aligner in self._config.filter_enabled_aligners
        if filter_applied:
            hits = [h for h in hits if h.score >= filter_threshold]

        return _CrossPlmChainResult(
            hits=hits,
            hits_found=len(order),
            hits_fetched=len(kept_ids),
            hits_pre_filter=hits_pre_filter,
            filter_applied=filter_applied,
            filter_threshold=filter_threshold,
            per_retrieval_stats=per_retrieval_stats,
        )

    async def _run_paired(
        self,
        *,
        query_seqs: list[str],
        query_ids: list[str],
        retrieval_models: list[str],
        resolve_collection: Any,
        score_model: str,
        k: int,
        aligner: str,
        mode: str,
        options: dict[str, Any],
        filter_by_score: bool,
    ) -> JobResult:
        """Paired-MSA pipeline via MMseqs-style taxonomy join.

        Per-chain retrieval runs in parallel with `paired_k = multiplier *
        effective_k` so the taxonomy-intersection step still leaves a
        useful pool per chain. Each chain's post-filter hits are then
        joined on the NCBI taxonomy id resolved from
        `tax:UniRef50_<acc>` (populated by `build_sequence_cache.py`).
        Rows where every chain shares a taxonomy emit one paired A3M
        record; the rest are dropped.

        Chains with an empty hit list short-circuit to a query-only A3M
        (the single-chain path already handles that for unpaired).
        """
        cfg = self._config
        paired_k = min(cfg.paired_k_multiplier * k, cfg.paired_k_cap)

        # Phase A — per-chain pipelines in parallel. Each chain sees its
        # own `cross_plm_hits` run; the retrieval + scoring topology is
        # identical to unpaired, only the per-chain `k` differs.
        chain_results = await asyncio.gather(
            *(
                self._cross_plm_hits(
                    query_seq=seq,
                    retrieval_models=retrieval_models,
                    resolve_collection=resolve_collection,
                    score_model=score_model,
                    k=paired_k,
                    aligner=aligner,
                    mode=mode,
                    options=options,
                    filter_by_score=filter_by_score,
                )
                for seq in query_seqs
            )
        )

        # Phase B — taxonomy lookup per chain. The fetcher owns the
        # `tax:*` keyspace; DictTargetFetcher returns {} by default so
        # in-memory tests degrade to an empty join (zero paired rows),
        # which the tests exercise explicitly.
        per_chain_tax: list[dict[str, str]] = []
        for cr in chain_results:
            ids = [h.target_id for h in cr.hits]
            per_chain_tax.append(await self._fetcher.fetch_taxonomy(ids))

        # Phase C — MMseqs-style join. Returns paired rows sorted by
        # joint score + bookkeeping counters for the stats block.
        join = join_by_taxonomy(
            [cr.hits for cr in chain_results],
            per_chain_tax,
        )

        # Phase D — assemble. Paired A3M always emits at least the
        # concatenated query even if no shared taxonomy survived.
        rows = [(r.taxonomy_id, r.hits, r.joint_score) for r in join.rows]
        a3m = assemble_paired_a3m(
            query_ids=query_ids,
            query_seqs=query_seqs,
            paired_rows=rows,
        )

        per_chain_stats: list[dict[str, Any]] = []
        for i, cr in enumerate(chain_results):
            per_chain_stats.append(
                {
                    "chain_index": i,
                    "hits_found": cr.hits_found,
                    "hits_fetched": cr.hits_fetched,
                    "hits_pre_filter": cr.hits_pre_filter,
                    "hits_post_filter": len(cr.hits),
                    "filter_applied": cr.filter_applied,
                    "filter_threshold": cr.filter_threshold,
                    "per_retrieval": cr.per_retrieval_stats,
                    "tax_resolved": join.per_chain_with_tax[i],
                    "taxonomies": join.taxonomies_per_chain[i],
                }
            )

        return JobResult(
            format="a3m",
            payload=a3m,
            stats={
                "pipeline": "orchestrator",
                "topology": "paired",
                "depth": 1 + len(join.rows),
                "paired_rows": len(join.rows),
                "paired_taxonomies": join.shared_taxonomies,
                "paired_k_multiplier": cfg.paired_k_multiplier,
                "paired_k_effective": paired_k,
                "retrieval_models": retrieval_models,
                "score_model": score_model,
                "aligner": aligner,
                "mode": mode,
                "filter_by_score": filter_by_score,
                "per_chain": per_chain_stats,
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
        self,
        http: httpx.AsyncClient,
        model: str,
        seqs: list[str],
        *,
        ids: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Batch `seqs` through the embedding service.

        When `ids` are known (target-embedding path in `_run_one_model`) and
        the model is in `shard_models`, the orchestrator tries the shard
        store first via `/embed_by_id`, then falls back to `/embed` for the
        misses. Otherwise the whole list goes through `/embed` in chunks of
        `embed_chunk_size` to bound peak GPU memory.

        The returned list is positionally aligned with the input `seqs` /
        `ids` so the caller's `kept_ids` ordering survives. That's what the
        downstream `/align` call expects (each target embedding at the same
        index as its target_id).
        """
        if not seqs:
            return []

        out: list[np.ndarray | None] = [None] * len(seqs)
        miss_positions: list[int] = list(range(len(seqs)))

        # Shard-first path: only when the caller gave us ids *and* the
        # model is opted in. No ids = query embedding or other pure-sequence
        # consumer — can't key into the shards.
        if ids is not None and model in self._config.shard_models:
            if len(ids) != len(seqs):
                raise ValueError("ids and seqs must be the same length")
            # /embed_by_id/bin returns compact binary frames that decode
            # zero-copy. At k=1000 this is ~150 s faster than the JSON
            # variant for the round-trip. Chunk size keeps each response
            # under a few hundred MB.
            shard_chunk = 1024
            found: dict[str, Any] = {}
            try:
                from plmmsa.align import binary as _binary

                for start in range(0, len(ids), shard_chunk):
                    chunk_ids = ids[start : start + shard_chunk]
                    resp = await http.post(
                        f"{self._config.embedding_url}/embed_by_id/bin",
                        json={"model": model, "ids": chunk_ids},
                    )
                    resp.raise_for_status()
                    meta, tensors = _binary.decode_tensors(resp.content)
                    for rid, t in zip(
                        meta.get("found_ids", []),
                        tensors,
                        strict=True,
                    ):
                        found[rid] = t
            except Exception:
                logger.warning(
                    "shard lookup failed for model=%s; falling back to /embed",
                    model,
                    exc_info=True,
                )
                found = {}
            hits: list[int] = []
            for i, rid in enumerate(ids):
                emb = found.get(rid)
                if emb is None:
                    continue
                out[i] = np.asarray(emb, dtype=np.float32)
                hits.append(i)
            miss_positions = [i for i in range(len(seqs)) if out[i] is None]
            if hits:
                logger.info(
                    "shard_store: %d/%d hits for model=%s (misses fall through to /embed)",
                    len(hits),
                    len(seqs),
                    model,
                )

        # Fallback path: anything not yet resolved goes through /embed, in
        # chunks that bound the GPU working set.
        if miss_positions:
            miss_seqs = [seqs[i] for i in miss_positions]
            fallback_embs = await self._embed_via_service(http, model, miss_seqs)
            for pos, arr in zip(miss_positions, fallback_embs, strict=True):
                out[pos] = arr

        if any(e is None for e in out):
            raise RuntimeError(
                f"target embedding left gaps for model={model}: "
                f"{sum(1 for e in out if e is None)}/{len(seqs)} unresolved"
            )
        return [e for e in out if e is not None]

    async def _embed_via_service(
        self, http: httpx.AsyncClient, model: str, seqs: list[str]
    ) -> list[np.ndarray]:
        chunk = max(1, self._config.embed_chunk_size)
        if len(seqs) <= 1:
            # Trivial case — no padding to save on.
            return await self._embed_chunks(http, model, seqs, chunk)

        # Sort descending by length so each batch's padding is set by
        # its longest member; short batches stay cheap. Track original
        # positions so the returned list is in caller-expected order.
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        sorted_seqs = [seqs[i] for i in order]
        sorted_out = await self._embed_chunks(http, model, sorted_seqs, chunk)
        # Restore caller's order.
        restored: list[np.ndarray | None] = [None] * len(seqs)
        for pos, orig_i in enumerate(order):
            restored[orig_i] = sorted_out[pos]
        return [x for x in restored if x is not None]

    async def _embed_chunks(
        self,
        http: httpx.AsyncClient,
        model: str,
        seqs: list[str],
        chunk: int,
    ) -> list[np.ndarray]:
        """Call the embedding service in GPU-sized chunks.

        When `align_transport == "binary"` (the default) we hit
        `/embed/bin`: request is still tiny JSON, response is binary-
        framed per-residue tensors that decode via `np.frombuffer`
        (~1-2 s for a 1.6 GB chunk vs. ~30 s for JSON).
        """
        use_binary = self._config.align_transport == "binary"
        if use_binary:
            from plmmsa.align import binary as _binary

        out: list[np.ndarray] = []
        for start in range(0, len(seqs), chunk):
            batch = seqs[start : start + chunk]
            if use_binary:
                resp = await http.post(
                    f"{self._config.embedding_url}/embed/bin",
                    json={"model": model, "sequences": batch},
                )
                resp.raise_for_status()
                _, tensors = _binary.decode_tensors(resp.content)
                out.extend(tensors)
            else:
                resp = await http.post(
                    f"{self._config.embedding_url}/embed",
                    json={"model": model, "sequences": batch},
                )
                resp.raise_for_status()
                body = resp.json()
                out.extend(np.asarray(e, dtype=np.float32) for e in body["embeddings"])
        return out

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
        if self._config.align_transport == "binary":
            from plmmsa.align import binary as align_binary

            body = align_binary.encode(
                {"aligner": aligner, "mode": mode, "options": dict(options)},
                query_emb,
                target_embs,
            )
            resp = await http.post(
                f"{self._config.align_url}/align/bin",
                content=body,
                headers={"Content-Type": align_binary.CONTENT_TYPE},
            )
            resp.raise_for_status()
            return resp.json()["alignments"]

        # JSON fallback — kept for tests and for back-compat with older
        # align images that don't have the binary endpoint.
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


@dataclass(slots=True)
class _CrossPlmChainResult:
    """Cross-PLM topology output for one chain.

    The single-chain wrapper (`_run_cross_plm`) turns this into a JobResult
    + A3M; the paired wrapper (`_run_paired`) collects one per chain and
    runs the taxonomy join over them.
    """

    hits: list[AlignmentHit]
    hits_found: int  # unioned neighbor count across retrieval VDBs
    hits_fetched: int  # target-sequence fetch survivors (pre-filter)
    hits_pre_filter: int
    filter_applied: bool
    filter_threshold: float
    per_retrieval_stats: dict[str, dict[str, Any]]


@dataclass(slots=True)
class _PerModelResult:
    """What one per-model pipeline run produced — merged in `Orchestrator.run`.

    Kept local to the module: it's not part of the worker→API contract; the
    `stats()` payload is what surfaces on the job record.
    """

    model: str
    collection: str
    hits_found: int
    hits: list[AlignmentHit]
    error: str | None = None

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "collection": self.collection,
            "hits_found": self.hits_found,
            "hits_fetched": len(self.hits),
        }
        if self.error is not None:
            out["error"] = self.error
        return out


def _query_only_result(
    *,
    query_id: str,
    query_seq: str,
    query_self_score: float,
    hits_found: int,
    hits_fetched: int,
    models: list[str],
    per_model_stats: dict[str, dict[str, Any]],
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
            "models": models,
            "per_model": per_model_stats,
            "note": "No target sequences could be fetched; MSA contains only the query.",
        },
    )
