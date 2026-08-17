#!/usr/bin/env bash
# Retrain every BabyLM SFT configuration previously evaluated, on the new
# magnitude-balanced split. Records run_ids to
# babylm_finetune/outputs/newsplit_sft_manifest.json
#
# Configs (10m + 100m each unless noted):
#   - all_types (full train.jsonl)
#   - only_constancy / only_intersection
#   - only_3easy (constancy,constancy_row,pattern)
#   - only_3hard (progression,combine,intersection)
#   - n_scaling N in {50,100,250,500,1000}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/newsplit_sft_manifest.json"
PILOT_DIR="babylm_finetune/data/sft/pilots/n_scaling"
TRAIN_FULL="babylm_finetune/data/sft/train.jsonl"
CFG10="babylm_finetune/configs/train/sft_babylm_10m.yaml"
CFG100="babylm_finetune/configs/train/sft_babylm_100m.yaml"
SIZES=(50 100 250 500 1000)

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$TRAIN_FULL" ]]; then
  echo "Missing $TRAIN_FULL — run generate_sft_splits.py first" >&2
  exit 1
fi
if [[ ! -f "$PILOT_DIR/train_n50.jsonl" ]]; then
  echo "Missing n-scaling pilots; run write_n_scaling_pilots.py first" >&2
  exit 1
fi

python3 - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({"runs": [], "split": "magnitude_balanced_20260731"}, indent=2) + "\n")
PY

record_run() {
  local config="$1"
  local run_name="$2"
  local train_file="$3"
  local include_types="${4:-}"
  local run_id="$5"
  python3 - <<PY
import json
from pathlib import Path
path = Path("$MANIFEST")
data = json.loads(path.read_text(encoding="utf-8"))
runs = data.setdefault("runs", [])
include = """$include_types""".strip() or None
entry = {
    "config": "$config",
    "run_name": "$run_name",
    "train_file": "$train_file",
    "include_types": include,
    "run_id": "$run_id",
    "model_tag": "$run_id".split("__", 1)[0],
}
runs[:] = [
    r for r in runs
    if not (
        r.get("model_tag") == entry["model_tag"]
        and r.get("run_name") == entry["run_name"]
    )
]
runs.append(entry)
data["split"] = "magnitude_balanced_20260731"
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {entry['run_id']} -> {path}")
PY
}

run_sft() {
  local config="$1"
  local run_name="$2"
  local train_file="$3"
  local include_types="${4:-}"

  # Skip if this (model_tag from config + run_name) is already recorded.
  local model_tag
  if [[ "$config" == *100m* ]]; then
    model_tag="babylm-100m-gpt2"
  else
    model_tag="babylm-10m-gpt2"
  fi
  if python3 - <<PY
import json
from pathlib import Path
path = Path("$MANIFEST")
data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
for r in data.get("runs") or []:
    if r.get("model_tag") == "$model_tag" and r.get("run_name") == "$run_name" and r.get("run_id"):
        print(r["run_id"])
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "=== SKIP $model_tag / $run_name (already in manifest) ==="
    return 0
  fi

  echo ""
  echo "=== SFT config=$config run_name=$run_name include=${include_types:-ALL} ==="
  local log
  log="$(mktemp)"
  local args=(
    modal run babylm_finetune/scripts/modal_sft.py
    --config "$config"
    --train-file "$train_file"
    --run-name "$run_name"
  )
  if [[ -n "$include_types" ]]; then
    args+=(--include-types "$include_types")
  fi
  "${args[@]}" | tee "$log"
  local run_id
  run_id="$(grep -E '^run_id: ' "$log" | tail -n1 | sed 's/^run_id: //')"
  rm -f "$log"
  if [[ -z "$run_id" ]]; then
    echo "Failed to parse run_id from modal output" >&2
    exit 1
  fi
  record_run "$config" "$run_name" "$train_file" "$include_types" "$run_id"
}

# --- Full / include-type configs (10m then 100m) ---
run_sft "$CFG10" "all_types" "$TRAIN_FULL"
run_sft "$CFG10" "10_only_constancy" "$TRAIN_FULL" "constancy"
run_sft "$CFG10" "10_only_intersection" "$TRAIN_FULL" "intersection"
run_sft "$CFG10" "10_only_3easy" "$TRAIN_FULL" "constancy,constancy_row,pattern"
run_sft "$CFG10" "10_only_3hard" "$TRAIN_FULL" "progression,combine,intersection"

run_sft "$CFG100" "all_types" "$TRAIN_FULL"
run_sft "$CFG100" "100_only_constancy" "$TRAIN_FULL" "constancy"
run_sft "$CFG100" "100_only_intersection" "$TRAIN_FULL" "intersection"
run_sft "$CFG100" "100_only_3easy" "$TRAIN_FULL" "constancy,constancy_row,pattern"
run_sft "$CFG100" "100_only_3hard" "$TRAIN_FULL" "progression,combine,intersection"

# --- N-scaling ---
for n in "${SIZES[@]}"; do
  run_sft "$CFG10" "n_scaling_n${n}" "$PILOT_DIR/train_n${n}.jsonl"
done
for n in "${SIZES[@]}"; do
  run_sft "$CFG100" "n_scaling_n${n}" "$PILOT_DIR/train_n${n}.jsonl"
done

echo ""
echo "=== New-split SFT training complete ==="
echo "Manifest: $MANIFEST"
echo "Next: bash babylm_finetune/scripts/eval_newsplit_sft.sh"
