#!/usr/bin/env bash
# Run ravens_numerical via vLLM (BabyLM GPT-2 baseline).
set -euo pipefail

MODEL="${MODEL:-BabyLM-community/babylm-baseline-10m-gpt2}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"

exec ravens-run \
  --backend vllm \
  --models "${MODEL}" \
  --tasks ravens_numerical \
  --n-examples 0 \
  --base-url "${VLLM_URL}" \
  "$@"
