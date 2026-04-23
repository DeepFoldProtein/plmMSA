FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        # gcc + build tooling: torch.compile / Inductor needs a C
        # compiler at runtime to build the Triton kernels that back
        # FlashSinkhorn. Without these, `_warmup_fused_sinkhorn` raises
        # "Failed to find C compiler" and falls back to the eager path.
        gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.11

WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock* ./
# OTalign needs torch for its Sinkhorn + cost-matrix (runs on the PLM
# device). Install the `align` extra so torch lives in the image; other
# aligners (PLMAlign, pLM-BLAST) are numba-only and don't need it.
RUN uv sync --extra align --no-install-project || uv sync --extra align

COPY src/ ./src/
RUN uv sync --extra align --no-editable

EXPOSE 8083

CMD ["uv", "run", "python", "-m", "plmmsa.align"]
