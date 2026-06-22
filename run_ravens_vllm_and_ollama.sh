#!/usr/bin/env bash
# Run ravens_numerical twice: Pythia via vLLM, then Qwen3 tags via Ollama (same defaults as
# run_pythia_ravens.sh + run_qwen3_ravens.sh). Start both servers first or use SKIP_VLLM / SKIP_OLLAMA=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}/baby_reasoning_eval/baby-reasoning"

runner() {
  if command -v uv >/dev/null 2>&1; then uv run script/run "$@"
  else python3 script/run "$@"; fi
}

PYTHIA="${PYTHIA_MODEL:-EleutherAI/pythia-70m-deduped}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODELS="${MODELS:-qwen3:30b qwen3:8b qwen3:4b qwen3:1.7b qwen3:0.6b}"

if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
  echo "=== ravens_numerical: vLLM / ${PYTHIA} ==="
  runner \
    --backend vllm \
    --base-url "${VLLM_URL}" \
    --models "${PYTHIA}" \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    "$@"
fi

if [[ "${SKIP_OLLAMA:-0}" != "1" ]]; then
  echo "=== ravens_numerical: Ollama / Qwen3 suite ==="
  runner \
    --backend ollama \
    --ollama-base-url "${OLLAMA_URL}" \
    --models ${MODELS} \
    --tasks ravens_numerical \
    --ravens-repo-root "${ROOT}" \
    --ravens-tasks-json "${ROOT}/tasks.json" \
    --n-examples 0 \
    "$@"
fi
