#!/usr/bin/env bash
# Local MiniBERTa eval (HuggingFace backend; no vLLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}"
MODEL="${MODEL:-nyu-mll/roberta-med-small-1M-1}"
PROMPT_TYPE="${PROMPT_TYPE:-instruction}"
N_EXAMPLES="${N_EXAMPLES:-0}"
MAX_TASKS="${MAX_TASKS:-}"
EXTRA=()
if [[ -n "${MAX_TASKS}" ]]; then
  EXTRA+=(--ravens-max-tasks "${MAX_TASKS}")
fi
exec python3 "${ROOT}/baby_reasoning_eval/baby-reasoning/script/run" \
  --backend hf \
  --models "${MODEL}" \
  --tasks ravens_numerical \
  --ravens-repo-root "${ROOT}" \
  --ravens-tasks-json "${ROOT}/tasks.json" \
  --n-examples "${N_EXAMPLES}" \
  --ravens-prompt-type "${PROMPT_TYPE}" \
  "${EXTRA[@]}"
