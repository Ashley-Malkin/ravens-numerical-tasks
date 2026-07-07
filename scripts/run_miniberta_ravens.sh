#!/usr/bin/env bash
# Run ravens_numerical via HuggingFace masked LM (MiniBERTa).
set -euo pipefail

MODEL="${MODEL:-nyu-mll/roberta-base-100M-1}"
MAX_TASKS="${MAX_TASKS:-}"

ARGS=(
  --backend hf
  --models "${MODEL}"
  --tasks ravens_numerical
  --n-examples 0
)
if [[ -n "${MAX_TASKS}" ]]; then
  ARGS+=(--ravens-max-tasks "${MAX_TASKS}")
fi
if [[ -n "${PROMPT_TYPE:-}" ]]; then
  ARGS+=(--ravens-prompt-type "${PROMPT_TYPE}")
fi

exec ravens-run "${ARGS[@]}" "$@"
