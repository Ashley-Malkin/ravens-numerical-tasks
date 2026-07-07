#!/usr/bin/env bash
# Run ravens_numerical via vLLM (Pythia by default).
# Start vLLM first, e.g.: vllm serve EleutherAI/pythia-70m-deduped --host 0.0.0.0 --port 8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-EleutherAI/pythia-70m-deduped}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"

exec ravens-run \
  --backend vllm \
  --models "${MODEL}" \
  --tasks ravens_numerical \
  --n-examples 0 \
  --base-url "${VLLM_URL}" \
  "$@"
