#!/usr/bin/env bash
# Eval N-scaling SFT run_ids from n_scaling_manifest.json into one combined log.
# Writes (with ``_n{N}`` from --n-examples, default 0):
#   babylm_finetune/logs/n_scaling_evals_n0.md
#   babylm_finetune/outputs/n_scaling_results.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/n_scaling_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST — run babylm_finetune/scripts/run_n_scaling_sft.sh first" >&2
  exit 1
fi

MODELS="$(python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runs = data.get("runs") or []
if not runs:
    raise SystemExit("n_scaling_manifest.json has no runs")
ids = [r["run_id"] for r in runs if r.get("run_id")]
if not ids:
    raise SystemExit("no run_id entries in manifest")
print(",".join(ids))
PY
)"

echo "Evaluating ${MODELS}"
exec modal run -m ravens_numerical.cloud.modal_eval \
  --models "$MODELS" \
  --n-examples 0 \
  --score-mode forced_choice \
  --combined-sft-log babylm_finetune/logs/n_scaling_evals.md \
  --combined-sft-results-json babylm_finetune/outputs/n_scaling_results.json \
  "$@"
