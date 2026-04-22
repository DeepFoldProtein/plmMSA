#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Copy .env.example to .env first and edit." >&2
    exit 1
fi

if [[ ! -f settings.toml ]]; then
    echo "Copy settings.example.toml to settings.toml first and edit." >&2
    exit 1
fi

docker compose up -d --build

set -a
# shellcheck disable=SC1091
source .env
set +a
: "${API_HOST_PORT:=8080}"

echo "Waiting for api to become healthy on port ${API_HOST_PORT}..."
for _ in $(seq 1 30); do
    if curl -sf -o /dev/null "http://localhost:${API_HOST_PORT}/health"; then
        echo "api is healthy."
        docker compose logs --tail=30 api
        exit 0
    fi
    sleep 3
done

echo "api did not become healthy in 90s. Run ./bin/logs.sh api for details." >&2
exit 2
