#!/usr/bin/env bash
# Run ravens_numerical via baby-reasoning + vLLM (BabyLM GPT-2 baseline by default).
# Start vLLM first, e.g.:
#   vllm serve BabyLM-community/babylm-baseline-10m-gpt2 --host 0.0.0.0 --port 8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-BabyLM-community/babylm-baseline-10m-gpt2}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"

cd "${ROOT}/baby_reasoning_eval/baby-reasoning"

if command -v uv >/dev/null 2>&1; then
  exec uv run script/run \
    --backend vllm \
    --models "${MODEL}" \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    --base-url "${VLLM_URL}" \
    "$@"
else
  exec python3 script/run \
    --backend vllm \
    --models "${MODEL}" \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    --base-url "${VLLM_URL}" \
    "$@"
fi
