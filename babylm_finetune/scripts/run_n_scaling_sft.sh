#!/usr/bin/env bash
# Run BabyLM N-scaling SFT: 10m + 100m × N in {50,100,250,500,1000}.
# Appends run records to babylm_finetune/outputs/n_scaling_manifest.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PILOT_DIR="babylm_finetune/data/sft/pilots/n_scaling"
MANIFEST="$ROOT/babylm_finetune/outputs/n_scaling_manifest.json"
SIZES=(50 100 250 500 1000)

mkdir -p "$(dirname "$MANIFEST")"

if [[ ! -f "$PILOT_DIR/train_n50.jsonl" ]]; then
  echo "Missing nested pilots; run: python3 babylm_finetune/scripts/write_n_scaling_pilots.py"
  exit 1
fi

python3 - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(json.dumps({"runs": []}, indent=2) + "\n", encoding="utf-8")
PY

run_one() {
  local config="$1"
  local n="$2"
  local train_file="$PILOT_DIR/train_n${n}.jsonl"
  local run_name="n_scaling_n${n}"
  echo ""
  echo "=== SFT config=$config N=$n ==="
  # Capture full modal output; parse final run_id line.
  local log
  log="$(mktemp)"
  modal run babylm_finetune/scripts/modal_sft.py \
    --config "$config" \
    --train-file "$train_file" \
    --run-name "$run_name" \
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
    "config": "$config",
    "N": $n,
    "run_name": "$run_name",
    "train_file": "$train_file",
    "run_id": run_id,
    "model_tag": run_id.split("__", 1)[0],
}
# Replace existing same (model_tag, N) if re-run.
runs[:] = [
    r for r in runs
    if not (r.get("model_tag") == entry["model_tag"] and r.get("N") == entry["N"])
]
runs.append(entry)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {entry['run_id']} -> {path}")
PY
}

for n in "${SIZES[@]}"; do
  run_one "babylm_finetune/configs/train/sft_babylm_10m.yaml" "$n"
done
for n in "${SIZES[@]}"; do
  run_one "babylm_finetune/configs/train/sft_babylm_100m.yaml" "$n"
done

echo ""
echo "=== N-scaling SFT complete ==="
echo "Manifest: $MANIFEST"
