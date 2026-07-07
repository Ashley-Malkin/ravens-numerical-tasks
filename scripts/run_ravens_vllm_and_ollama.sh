#!/usr/bin/env bash
# Run Pythia (vLLM) then Qwen3 (Ollama) ravens_numerical evals in sequence.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${SKIP_VLLM:-}" ]]; then
  "${ROOT}/run_pythia_ravens.sh" "$@"
fi

if [[ -z "${SKIP_OLLAMA:-}" ]]; then
  "${ROOT}/run_qwen3_ravens.sh" "$@"
fi
