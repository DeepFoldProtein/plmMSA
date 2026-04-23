"""ColabFold-compatible MSA server endpoints.

Translates the MMseqs2 MsaServer wire shape that `colabfold_batch
--host-url`, `boltz predict --msa_server_url`, and Protenix's
MSA-server config key all speak, onto the existing plmMSA job
lifecycle (`/v2/msa{,/{id}}`). The factory builds one router per
aligner flavor so clients pick PLMAlign vs. OTalign via the URL
path (CF's form shape cannot carry that choice, and hardwiring one
flavor per prefix keeps the mapping obvious).

Expected wiring in `plmmsa.api.__init__`:

    from plmmsa.api.routes.colabfold import make_router
    app.include_router(make_router("plmalign"),
                       prefix="/v2/colabfold/plmmsa", tags=["colabfold"])
    app.include_router(make_router("otalign"),
                       prefix="/v2/colabfold/otalign", tags=["colabfold"])

Ticket id is the plmMSA job id verbatim — no second storage, no
translation table. CF clients treat the id as opaque so the UUID
shape doesn't matter to them.

CF's MsaServer reference: https://github.com/sokrypton/ColabFold/tree/main/MsaServer
"""

from __future__ import annotations

import io
import logging
import tarfile
from typing import Any

from fastapi import APIRouter, Form, Path, Request
from fastapi.responses import JSONResponse, StreamingResponse

from plmmsa.api.middleware import audit_event
from plmmsa.api.routes import v2 as v2_mod
from plmmsa.api.routes.v2 import (
    _IDEMPOTENCY_TTL_S,
    SubmitRequest,
    _enforce_backpressure,
    _idempotency_key,
    _resolve_collections_for_models,
    _resolve_query_ids,
    _validate_submit,
)
from plmmsa.config import get_settings
from plmmsa.errors import ErrorCode, PlmMSAError
from plmmsa.jobs.models import JobStatus

logger = logging.getLogger(__name__)

# MsaServer returns one of these status strings; everything else is a
# protocol error from CF's point of view.
_STATUS_MAP: dict[JobStatus, str] = {
    JobStatus.QUEUED: "PENDING",
    JobStatus.RUNNING: "RUNNING",
    JobStatus.SUCCEEDED: "COMPLETE",
    JobStatus.FAILED: "ERROR",
    JobStatus.CANCELLED: "ERROR",
}


def _parse_query_body(q: str) -> str:
    """Accept either a raw amino-acid string or a tiny FASTA record.
    Returns a single sequence with whitespace stripped. CF clients
    typically send FASTA with one `>header` + the sequence on
    subsequent lines.
    """
    text = q.strip()
    if not text:
        raise PlmMSAError(
            "ColabFold: `q` is empty.",
            code=ErrorCode.INVALID_FASTA,
            http_status=400,
        )
    if text.startswith(">"):
        # Drop the header line(s); join the rest. We only serve a
        # single-chain MSA here — paired-multimer goes through
        # /ticket/pair.
        lines = text.splitlines()
        seq = "".join(ln.strip() for ln in lines if not ln.startswith(">"))
        if not seq:
            raise PlmMSAError(
                "ColabFold: FASTA record had no sequence lines.",
                code=ErrorCode.INVALID_FASTA,
                http_status=400,
            )
        return seq
    # Raw sequence — drop whitespace just in case the client wrapped it.
    return "".join(text.split())


def _cf_tar(a3m: str) -> bytes:
    """Wrap an A3M string in the tar layout CF expects. Three entries:
    `uniref.a3m` (unpaired), `pair.a3m` (empty — paired MSA is a
    separate endpoint), `bfd.mgnify30.metaeuk30.smag30.a3m` (CF treats
    this as a secondary MSA source; we serve the plmMSA A3M under it
    too so clients that union both still work).
    """
    buf = io.BytesIO()
    payload_bytes = a3m.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in (
            ("uniref.a3m", payload_bytes),
            ("bfd.mgnify30.metaeuk30.smag30.a3m", payload_bytes),
            ("pair.a3m", b""),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def make_router(aligner_id: str) -> APIRouter:
    """Factory — one router per ColabFold flavor.

    The aligner id is baked into every `/ticket/msa` submission from
    this router. Status / download handlers are aligner-agnostic
    (they just look up the job by id), so the two registered routers
    share those handlers without interfering.
    """
    router = APIRouter()

    @router.post("/ticket/msa")
    async def ticket_msa(
        request: Request,
        q: str = Form(...),
        mode: str | None = Form(None),
        database: str | None = Form(None),
    ) -> JSONResponse:
        # CF's `mode` is things like "env-pairgreedy"; we don't expose
        # that surface yet. Accept and ignore. `database` is
        # "uniref"/"envdb"; we always retrieve from our aggregate VDBs.
        if database and database not in ("uniref", "", None):
            logger.info(
                "colabfold: non-default database=%r ignored (aligner=%s)",
                database,
                aligner_id,
            )
        sequence = _parse_query_body(q)

        # Build our native SubmitRequest with the flavor's aligner
        # hardwired. Everything else (models, score_model, filter) takes
        # the server defaults — same surface a caller hitting /v2/msa
        # with no overrides would get.
        settings = get_settings()
        # Validate via model_validate so pyright sees the optional-field
        # defaults; direct kwarg construction trips the type-checker
        # because Pydantic v2's Field(None, ...) isn't recognized as
        # Optional without the model-validate path.
        submit_req = SubmitRequest.model_validate({"sequences": [sequence], "aligner": aligner_id})
        resolved_models, resolved_score = _validate_submit(submit_req, settings)
        store = await v2_mod._get_job_store()
        token_id = getattr(request.state, "token_id", None)
        request_id = getattr(request.state, "request_id", None)
        client_ip = getattr(request.state, "client_ip", None)

        payload: dict[str, Any] = submit_req.model_dump()
        payload["models"] = resolved_models
        payload["score_model"] = resolved_score
        payload["query_ids"] = _resolve_query_ids(submit_req)
        payload.pop("query_id", None)
        payload["collections"] = _resolve_collections_for_models(
            resolved_models,
            settings,
        )
        payload.pop("model", None)
        # Tag the record so audit trails can tell CF-originated jobs
        # apart from native /v2/msa submissions at a glance.
        payload["_cf_flavor"] = aligner_id

        idem_key = _idempotency_key(token_id or client_ip, payload)
        prior = await store.redis.get(idem_key)
        if prior is not None:
            prior_id = prior.decode() if isinstance(prior, bytes) else prior
            prior_job = await store.get(prior_id)
            if prior_job is not None:
                audit_event(
                    "colabfold.submit.idempotent",
                    token_id=token_id,
                    request_id=request_id,
                    client_ip=client_ip,
                    job_id=prior_id,
                    flavor=aligner_id,
                )
                return JSONResponse(
                    content={
                        "id": prior_id,
                        "status": _STATUS_MAP.get(prior_job.status, "UNKNOWN"),
                    }
                )

        await _enforce_backpressure(store, settings)
        job = await store.create(payload)
        await store.redis.set(idem_key, job.id, ex=_IDEMPOTENCY_TTL_S)  # pyright: ignore[reportGeneralTypeIssues]
        audit_event(
            "colabfold.submit",
            token_id=token_id,
            request_id=request_id,
            client_ip=client_ip,
            job_id=job.id,
            flavor=aligner_id,
            chain_len=len(sequence),
        )
        return JSONResponse(content={"id": job.id, "status": "PENDING"})

    @router.post("/ticket/pair")
    async def ticket_pair() -> JSONResponse:
        # Paired-MSA path isn't wired yet. Keep the route discoverable
        # and return a CF-shaped error body so the client surfaces a
        # readable message rather than "unexpected 404".
        return JSONResponse(
            status_code=501,
            content={
                "status": "ERROR",
                "message": (
                    "plmMSA paired MSA is not implemented yet. "
                    "Use /ticket/msa for single-chain MSAs."
                ),
            },
        )

    @router.get("/ticket/msa/{ticket}")
    async def ticket_status(ticket: str = Path(..., min_length=1)) -> JSONResponse:
        store = await v2_mod._get_job_store()
        job = await store.get(ticket)
        if job is None:
            return JSONResponse(content={"id": ticket, "status": "UNKNOWN"})
        return JSONResponse(
            content={
                "id": ticket,
                "status": _STATUS_MAP.get(job.status, "UNKNOWN"),
            }
        )

    @router.get("/result/download/{ticket}")
    async def ticket_download(ticket: str = Path(..., min_length=1)) -> StreamingResponse:
        store = await v2_mod._get_job_store()
        job = await store.get(ticket)
        if job is None or job.result is None:
            # Match CF's content-type so the client's tar-extract code
            # surfaces a clear error (rather than a mysterious 404 body).
            raise PlmMSAError(
                f"Ticket {ticket} has no result payload.",
                code=ErrorCode.JOB_NOT_FOUND,
                http_status=404,
                detail={"ticket": ticket},
            )
        tar_bytes = _cf_tar(job.result.payload)
        return StreamingResponse(
            io.BytesIO(tar_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{ticket}.tar"'},
        )

    return router


__all__ = ["make_router"]
