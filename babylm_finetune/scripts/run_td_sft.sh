#!/usr/bin/env bash
# SFT all TinyDialogues GPT-2 budgets (seed42: 10M / 20M / 50M / 100M / 200M)
# on the same Raven train/val JSONL used for BabyLM / CHILDES. Records run_ids to
# babylm_finetune/outputs/td_sft_manifest.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/td_sft_manifest.json"
CFG="babylm_finetune/configs/train/sft_td.yaml"
TRAIN_FULL="babylm_finetune/data/sft/train.jsonl"
BUDGETS=(10M 20M 50M 100M 200M)

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$TRAIN_FULL" ]]; then
  echo "Missing $TRAIN_FULL — run generate_sft_splits.py first" >&2
  exit 1
fi

python3 - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(
        json.dumps(
            {
                "runs": [],
                "source": "tinydialogues",
                "drive_parent_id": "1G2jw_WbJXRTbx825oMX_5nePcBnP_ZsZ",
                "seed": 42,
                "budgets": ["10M", "20M", "50M", "100M", "200M"],
                "train_file": "babylm_finetune/data/sft/train.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
PY

run_one() {
  local budget="$1"
  local model="td-${budget}"
  local run_name="all_types"
  echo ""
  echo "=== TinyDialogues SFT budget=${budget} model=${model} ==="
  local log
  log="$(mktemp)"
  modal run babylm_finetune/scripts/modal_sft.py \
    --config "$CFG" \
    --model "$model" \
    --train-file train \
    --run-name "$run_name" \
    --seed 42 \
    | tee "$log"
  local run_id
  run_id="$(grep -E '^run_id: ' "$log" | tail -n1 | sed 's/^run_id: //')"
  rm -f "$log"
  if [[ -z "$run_id" ]]; then
    echo "Failed to parse run_id from modal output" >&2
    exit 1
  fi
  python3 - <<PY
import json
from pathlib import Path
path = Path("$MANIFEST")
data = json.loads(path.read_text(encoding="utf-8"))
runs = data.setdefault("runs", [])
run_id = "$run_id"
entry = {
    "budget": "$budget",
    "model": "td-$budget",
    "config": "$CFG",
    "run_name": "$run_name",
    "train_file": "$TRAIN_FULL",
    "run_id": run_id,
    "model_tag": run_id.split("__", 1)[0],
    "seed": 42,
}
runs[:] = [r for r in runs if r.get("budget") != entry["budget"]]
runs.append(entry)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {entry['run_id']} -> {path}")
PY
}

for budget in "${BUDGETS[@]}"; do
  run_one "$budget"
done

echo ""
echo "Done. Manifest: $MANIFEST"
