#!/usr/bin/env bash
# Eval all new-split SFT run_ids into babylm_finetune/logs/newsplitSFT_n{N}.md
# (``--combined-sft-log`` base name; modal_eval appends ``_n{N}`` from --n-examples).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/newsplit_sft_manifest.json"
OUT_MD="babylm_finetune/logs/newsplitSFT.md"
OUT_JSON="babylm_finetune/outputs/newsplit_sft_results.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST — run babylm_finetune/scripts/run_newsplit_sft.sh first" >&2
  exit 1
fi

MODELS="$(python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runs = data.get("runs") or []
if not runs:
    raise SystemExit("newsplit_sft_manifest.json has no runs")
ids = [r["run_id"] for r in runs if r.get("run_id")]
if not ids:
    raise SystemExit("no run_id entries in manifest")
print(",".join(ids))
print(f"Evaluating {len(ids)} models", file=sys.stderr)
PY
)"

echo "Evaluating models → ${OUT_MD%.md}_n<N>.md (N from --n-examples)"
exec modal run -m ravens_numerical.cloud.modal_eval \
  --models "$MODELS" \
  --n-examples 0 \
  --score-mode forced_choice \
  --combined-sft-log "$OUT_MD" \
  --combined-sft-results-json "$OUT_JSON" \
  "$@"
