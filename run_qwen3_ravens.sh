#!/usr/bin/env bash
# Run ravens_numerical via Ollama (Qwen3 tags aligned with evaluate.py DEBUG_MODELS).
# Requires: ollama serve / models pulled (e.g. ollama pull qwen3:8b).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
# Same list as evaluate.DEBUG_MODELS / ravens_eval_models.QWEN3_OLLAMA_MODELS
MODELS="${MODELS:-qwen3:30b qwen3:8b qwen3:4b qwen3:1.7b qwen3:0.6b}"

cd "${ROOT}/baby_reasoning_eval/baby-reasoning"

if command -v uv >/dev/null 2>&1; then
  exec uv run script/run \
    --backend ollama \
    --ollama-base-url "${OLLAMA_URL}" \
    --models ${MODELS} \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    "$@"
else
  exec python3 script/run \
    --backend ollama \
    --ollama-base-url "${OLLAMA_URL}" \
    --models ${MODELS} \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    "$@"
fi
