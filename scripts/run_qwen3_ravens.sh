#!/usr/bin/env bash
# Run ravens_numerical via Ollama (Qwen3 by default).
set -euo pipefail

MODEL="${MODEL:-qwen3:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

exec ravens-run \
  --backend ollama \
  --models "${MODEL}" \
  --tasks ravens_numerical \
  --n-examples 0 \
  --ollama-base-url "${OLLAMA_URL}" \
  "$@"
