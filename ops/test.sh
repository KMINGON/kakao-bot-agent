#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f ops/compose.test.yaml up -d --wait
trap 'docker compose -f ops/compose.test.yaml down --volumes' EXIT
export TEST_DATABASE_URL='postgresql://kakao_test:disposable-test-only@127.0.0.1:15432/kakao_test'
uv run --frozen pytest "$@"
