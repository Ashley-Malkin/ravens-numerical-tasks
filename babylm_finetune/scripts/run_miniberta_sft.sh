#!/usr/bin/env bash
# MLM-finetune MiniBERTa (seed 1) at all corpus sizes (1M / 10M / 100M / 1B)
# on the same Raven train/val JSONL. Records run_ids to
# babylm_finetune/outputs/miniberta_sft_manifest.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/miniberta_sft_manifest.json"
CFG="babylm_finetune/configs/train/sft_miniberta.yaml"
TRAIN_FULL="babylm_finetune/data/sft/train.jsonl"
SIZES=(1M 10M 100M 1B)

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$TRAIN_FULL" ]]; then
  echo "Missing $TRAIN_FULL — run generate_sft_splits.py first" >&2
  exit 1
fi

echo "Checking train/val/test overlaps…"
python3 babylm_finetune/scripts/check_overlaps.py

python3 - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    path.write_text(
        json.dumps(
            {
                "runs": [],
                "source": "miniberta",
                "trainer": "mlm",
                "seed": 1,
                "sizes": ["1M", "10M", "100M", "1B"],
                "train_file": "babylm_finetune/data/sft/train.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
PY

run_one() {
  local size="$1"
  local model="miniberta-${size}"
  local run_name="all_types"
  echo ""
  echo "=== MiniBERTa MLM SFT size=${size} model=${model} ==="
  local log
  log="$(mktemp)"
  modal run babylm_finetune/scripts/modal_sft.py \
    --config "$CFG" \
    --model "$model" \
    --train-file train \
    --run-name "$run_name" \
    --seed 42 \
    --trainer mlm \
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
    "size": "$size",
    "model": "miniberta-$size",
    "hub_seed": 1,
    "config": "$CFG",
    "run_name": "$run_name",
    "train_file": "$TRAIN_FULL",
    "run_id": run_id,
    "model_tag": run_id.split("__", 1)[0],
    "train_seed": 42,
    "trainer": "mlm",
}
runs[:] = [r for r in runs if r.get("size") != entry["size"]]
runs.append(entry)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {entry['run_id']} -> {path}")
PY
}

for size in "${SIZES[@]}"; do
  run_one "$size"
done

echo ""
echo "Done. Manifest: $MANIFEST"
