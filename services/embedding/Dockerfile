FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.11

WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock* ./
RUN uv sync --extra embedding --no-install-project || uv sync --extra embedding

COPY src/ ./src/
RUN uv sync --extra embedding --no-editable

EXPOSE 8081

CMD ["uv", "run", "python", "-m", "plmmsa.embedding"]
