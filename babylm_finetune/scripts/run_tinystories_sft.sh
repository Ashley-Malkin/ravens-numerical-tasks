#!/usr/bin/env bash
# SFT TinyStories GPT-2-small 10M on the same Raven train/val JSONL used for
# BabyLM / CHILDES / TD. Records run_id to
# babylm_finetune/outputs/tinystories_sft_manifest.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MANIFEST="$ROOT/babylm_finetune/outputs/tinystories_sft_manifest.json"
CFG="babylm_finetune/configs/train/sft_tinystories.yaml"
TRAIN_FULL="babylm_finetune/data/sft/train.jsonl"
MODEL="ts"
RUN_NAME="all_types"

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
                "source": "tinystories",
                "budget": "10M",
                "train_file": "babylm_finetune/data/sft/train.jsonl",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
PY

echo ""
echo "=== TinyStories SFT model=${MODEL} ==="
log="$(mktemp)"
modal run babylm_finetune/scripts/modal_sft.py \
  --config "$CFG" \
  --model "$MODEL" \
  --train-file train \
  --run-name "$RUN_NAME" \
  --seed 42 \
  | tee "$log"
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
    "budget": "10M",
    "model": "$MODEL",
    "config": "$CFG",
    "run_name": "$RUN_NAME",
    "train_file": "$TRAIN_FULL",
    "run_id": run_id,
    "model_tag": run_id.split("__", 1)[0],
    "train_seed": 42,
}
runs[:] = [r for r in runs if r.get("budget") != entry["budget"]]
runs.append(entry)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Recorded {entry['run_id']} -> {path}")
PY

echo ""
echo "Done. Manifest: $MANIFEST"
