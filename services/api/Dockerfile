FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.11

WORKDIR /app

# Dependency install layer — copy only manifests so Docker caches this across
# source edits. uv.lock is optional on the first build; once generated it
# should be committed and used here with --frozen.
COPY pyproject.toml README.md ./
COPY uv.lock* ./
RUN uv sync --no-install-project || uv sync

COPY src/ ./src/
RUN uv sync --no-editable

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "plmmsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
