#!/usr/bin/env bash
# Eval every unique BabyLM SFT Volume checkpoint on the shared Raven test set.
# Writes babylm_finetune/logs/all_sft_evals_n{N}.md (run_id → overall / by-task).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec modal run -m ravens_numerical.cloud.modal_eval \
  --models babylm-sft \
  --n-examples 0 \
  --score-mode forced_choice \
  "$@"
