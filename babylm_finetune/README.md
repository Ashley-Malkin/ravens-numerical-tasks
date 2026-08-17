# BabyLM finetune — SFT data + training

0-shot completion SFT for Raven’s numerical tasks on BabyLM GPT-2 baselines
(`10m` / `100m`). Train / val / test are generated together with **no matrix
overlap**. Scalar task types (`constancy`, `constancy_row`, `pattern`,
`progression`, `distribution_of_three`, `progression_plus_n`, `tuple_grid`) are
**50/50 small vs large** in train/val (small = answer `1`–`250`, large =
`251`–`500`). Digit/tuple types (`combine`, `intersection`, `pattern_tuple`)
stay in `0`–`9`. Test is **50/type (500)** and synced to
[`data/complete.json`](../data/complete.json). [`data/tasks.json`](../data/tasks.json)
is the frozen 7-type 350-item backup.

## Generate splits

From the repo root (with `ravens_numerical` importable):

```bash
python3 babylm_finetune/scripts/generate_sft_splits.py
python3 babylm_finetune/scripts/check_overlaps.py
```

Useful flags: `--seed`, `--pilot-sizes 10,50,100`, `--out-dir`,
`--min` / `--max` / `--small-max` (default bands `1`–`250` and `251`–`500`).

After regenerating, re-run any holdout filters (`select_task_types.py`) and
N-scaling pilots if you use them — they are derived from `train.jsonl`.

## Retrain + eval on new split

Redo every previously tested SFT configuration on the magnitude-balanced data,
then write one combined eval file:

```bash
bash babylm_finetune/scripts/run_newsplit_sft.sh
bash babylm_finetune/scripts/eval_newsplit_sft.sh
# → babylm_finetune/logs/newsplitSFT_n0.md (suffix from --n-examples)
```

Configs covered (10m + 100m): `all_types`, `only_constancy`, `only_intersection`,
`only_3easy`, `only_3hard`, and n-scaling `N∈{50,100,250,500,1000}`.

## Data outputs

| Path | Description |
|------|-------------|
| `data/test.json` | Eval set (50/type = 500; synced with `data/complete.json`) |
| `data/oneICL_{train,val,test}.json` | Same split tasks; each task has unique 1-shot ICL |
| `data/threeICL_{train,val,test}.json` | Same split tasks; each task has unique 3-shot ICL |
| `data/sft/oneICL_{train,val}.jsonl` | ICL SFT JSONL (per-task 1 demo); `--train-file oneICL` |
| `data/sft/threeICL_{train,val}.jsonl` | ICL SFT JSONL (per-task 3 demos); `--train-file threeICL` |
| `data/train.json` / `data/val.json` | Task JSON (magnitude-balanced) |
| `data/sft/train.jsonl` / `val.jsonl` | 0-shot `{prompt, completion, task_type, ...}` |
| `data/sft/pilots/train_n{N}.jsonl` | Stratified pilots |
| `data/sft/pilots/n_scaling/train_n{N}.jsonl` | Nested N-scaling pilots |

Per magnitude type, quotas are train **90/90**, val **10/10**, test **25/25**
(small/large).

## Task-type holdout

```bash
python3 babylm_finetune/scripts/select_task_types.py \
  --holdout-types progression \
  --run-name holdout_progression
```

Or `--include-types constancy,pattern,...` / YAML under `configs/`. Writes
`data/sft/runs/<run_name>/`. Never filters `test.json`.

## Training (Modal — primary)

Install once: `pip install -r babylm_finetune/requirements-train.txt` and `modal setup`.

```bash
# Pilot 10m (alias or full path)
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_babylm_10m.yaml \
  --train-file train_n50

# Full train with 1-shot or 3-shot ICL demos in each prompt
# (auto-pairs matching oneICL_val / threeICL_val JSONL)
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_babylm_10m.yaml \
  --train-file oneICL \
  --run-name oneICL

modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_babylm_10m.yaml \
  --train-file threeICL \
  --run-name threeICL

# Full 100m, hold out progression
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_babylm_100m.yaml \
  --holdout-types progression \
  --run-name holdout_progression
```

``--train-file`` aliases: `train`, `train_n10` / `train_n50` / `train_n100`,
`oneICL` / `oneICL_train`, `threeICL` / `threeICL_train` (or any JSONL path).
Build ICL JSON + JSONL with `python3 babylm_finetune/scripts/build_icl_test_sets.py`.

Defaults: LR **`1e-4`**, completion-only loss, TensorBoard (no W&B).

### CHILDES ladder (seed42)

Same Raven SFT data and hypers as BabyLM 10m, for
`mcxfrank/childes-gpt2-ladder` budgets **1M / 5M / 12M / 24M** at pretraining
seed **42** (subfolders `development/seed42/rung{B}`).

```bash
# Base eval (all four budgets)
modal run -m ravens_numerical.cloud.modal_eval \
  --models childes --n-examples 0 --score-mode forced_choice
# → artifacts/logs/childesExperiments.md

# Or a single budget: --models childes-1M

# SFT all four on the same train.jsonl / val.jsonl
bash babylm_finetune/scripts/run_childes_sft.sh
# → babylm_finetune/outputs/childes_sft_manifest.json

# Eval one SFT checkpoint (run_id from the manifest)
modal run -m ravens_numerical.cloud.modal_eval \
  --models childes-seed42-rung1M__all_types__<timestamp> \
  --n-examples 0 --score-mode forced_choice

# Eval all saved CHILDES SFT checkpoints (run_id starts with childes-)
modal run -m ravens_numerical.cloud.modal_eval \
  --models childes-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/childes_sft_evals_n0.md
```

Single-budget SFT:

```bash
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_childes.yaml \
  --model childes-5M \
  --train-file train \
  --run-name all_types \
  --seed 42
```

### TinyDialogues GPT-2 (seed42)

Same Raven SFT data and hypers as BabyLM 10m / CHILDES, for final
`GPT2-small_TD_{10,20,50,100,200}M_20-epochs_seed42` checkpoints from
[Google Drive](https://drive.google.com/drive/folders/1G2jw_WbJXRTbx825oMX_5nePcBnP_ZsZ)
(ignores int-ckpt / CHILDES / TD_data / tokenizers / Test Model).

```bash
# Once: put finals on Modal Volume ravens-td-base
# Drive often blocks unauthenticated gdown — prefer browser download, then:
modal run babylm_finetune/scripts/upload_td_base_volume.py \
  --from-local /path/to/folder_with_GPT2-small_TD_*_finals
# (Only the five final folders; ignore int-ckpt / TD_data / tokenizers.)

# Or, if the Drive folder is shared as "Anyone with the link":
# modal run babylm_finetune/scripts/upload_td_base_volume.py

# Base eval (all five budgets)
modal run -m ravens_numerical.cloud.modal_eval \
  --models td --n-examples 0 --score-mode forced_choice
# → artifacts/logs/tdExperiments.md

# Or a single budget: --models td-10M

# SFT all five on the same train.jsonl / val.jsonl
bash babylm_finetune/scripts/run_td_sft.sh
# → babylm_finetune/outputs/td_sft_manifest.json

# Eval all saved TinyDialogues SFT checkpoints (run_id starts with td-)
modal run -m ravens_numerical.cloud.modal_eval \
  --models td-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/td_sft_evals_n0.md
```

Single-budget SFT:

```bash
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_td.yaml \
  --model td-50M \
  --train-file train \
  --run-name all_types \
  --seed 42
```

### OpenSubtitles GPT-2 (20M, seeds 0/42/123)

Same Raven SFT data and hypers as TD / CHILDES. Local finals live under
`open-subtitles-models/GPT2-small_opensubtitles_20M_…_seed{S}`.

```bash
# Once: put finals on Modal Volume ravens-os-base
modal run babylm_finetune/scripts/upload_opensubtitles_base_volume.py

# Base eval (all three seeds)
modal run -m ravens_numerical.cloud.modal_eval \
  --models os --n-examples 0 --score-mode forced_choice
# → artifacts/logs/opensubtitlesExperiments.md

# SFT all three seeds
bash babylm_finetune/scripts/run_opensubtitles_sft.sh
# → babylm_finetune/outputs/opensubtitles_sft_manifest.json

# Eval OpenSubtitles SFT (run_id starts with os-)
modal run -m ravens_numerical.cloud.modal_eval \
  --models os-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/os_sft_evals_n0.md
```

### Wiki GPT-2 (20M, seeds 0/42/123)

Same pattern; local finals under `wiki-models/GPT2-small_wiki_20M_…_seed{S}`.

```bash
modal run babylm_finetune/scripts/upload_wiki_base_volume.py

modal run -m ravens_numerical.cloud.modal_eval \
  --models wiki --n-examples 0 --score-mode forced_choice
# → artifacts/logs/wikiExperiments.md

bash babylm_finetune/scripts/run_wiki_sft.sh
# → babylm_finetune/outputs/wiki_sft_manifest.json

modal run -m ravens_numerical.cloud.modal_eval \
  --models wiki-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/wiki_sft_evals_n0.md
```

### TinyStories GPT-2 (10M)

Same Raven SFT data and hypers as TD / OS / Wiki. Local final lives under
`tiny-stories-model/GPT2-small_tinystories_10m_1e-04`.

```bash
# Once: put final on Modal Volume ravens-ts-base
modal run babylm_finetune/scripts/upload_tinystories_base_volume.py

# Base eval
modal run -m ravens_numerical.cloud.modal_eval \
  --models ts --n-examples 0 --score-mode forced_choice
# → artifacts/logs/tinystoriesExperiments.md

# SFT
bash babylm_finetune/scripts/run_tinystories_sft.sh
# → babylm_finetune/outputs/tinystories_sft_manifest.json

# Eval TinyStories SFT (run_id starts with ts-)
modal run -m ravens_numerical.cloud.modal_eval \
  --models ts-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/ts_sft_evals_n0.md
```

### MiniBERTa (seed 1, all corpus sizes)

MLM finetune (`AutoModelForMaskedLM` + HF `Trainer`, ~15% mask) on the same
Raven `train.jsonl` / `val.jsonl`. Bases download from Hub (no base Volume).
Eval uses HF PLL forced_choice — not vLLM.

Aliases: `miniberta-1M`, `miniberta-10M`, `miniberta-100M`, `miniberta-1B`
→ seed-1 Hub ids. Tags: `miniberta-{size}-seed1`.

```bash
# Assert train/val/test still disjoint
python babylm_finetune/scripts/check_overlaps.py

# Base eval (Hub; all 12 seeds×sizes via --models miniberta, or one size)
modal run -m ravens_numerical.cloud.modal_eval \
  --models miniberta-10M --n-examples 0 --score-mode forced_choice

# MLM SFT all sizes at seed 1
bash babylm_finetune/scripts/run_miniberta_sft.sh
# → babylm_finetune/outputs/miniberta_sft_manifest.json

# Eval MiniBERTa Volume SFT (run_id starts with miniberta-)
modal run -m ravens_numerical.cloud.modal_eval \
  --models miniberta-sft --n-examples 0 --score-mode forced_choice
# → babylm_finetune/logs/miniberta_sft_evals_n0.md
```

Single-size MLM SFT:

```bash
modal run babylm_finetune/scripts/modal_sft.py \
  --config babylm_finetune/configs/train/sft_miniberta.yaml \
  --model miniberta-10M \
  --train-file train \
  --run-name all_types \
  --trainer mlm \
  --seed 42
```

### Volumes and `run_id`

| Volume | Mount | Role |
|--------|-------|------|
| `ravens-hf-cache` | `/root/.cache/huggingface` | Base BabyLM / CHILDES downloads |
| `ravens-td-base` | `/td-base` | TinyDialogues base GPT-2 checkpoints |
| `ravens-os-base` | `/os-base` | OpenSubtitles base GPT-2 checkpoints |
| `ravens-wiki-base` | `/wiki-base` | Wiki base GPT-2 checkpoints |
| `ravens-ts-base` | `/ts-base` | TinyStories base GPT-2 checkpoint |
| `ravens-babylm-sft` | `/checkpoints` | Finetuned runs (source of truth) |

`run_id` format: `{model_tag}__{run_name}__{utc_timestamp}`  
Example: `babylm-10m-gpt2__holdout_progression__20260723T210501Z`

On-volume layout:

```text
/checkpoints/<run_id>/
  manifest.json
  resolved_config.yaml
  train.log
  tb/
  checkpoint-*/
  (final model + tokenizer)
```

The Modal job prints `run_id` and mirrors `manifest.json` / `train_summary.json` to
local `babylm_finetune/outputs/<run_id>/`.

## Eval (Modal)

Same stack as Hub BabyLM: `modal_eval` + vLLM, `task_type=ravens`,
`prompt_type=completion`, `score_mode=forced_choice` on `data/complete.json`
(SFT `test.json`). Pass a bare `run_id` or `/checkpoints/<run_id>`; weights load
from Volume `ravens-babylm-sft`.

```bash
modal run -m ravens_numerical.cloud.modal_eval \
  --models babylm-10m-gpt2__all_types__20260723T212815Z \
  --n-examples 0 \
  --score-mode forced_choice
```

Eval all BabyLM SFT checkpoints on the Volume (`run_id` starts with `babylm-`),
writing one combined file with each `run_id` above its scores:

```bash
modal run -m ravens_numerical.cloud.modal_eval \
  --models babylm-sft \
  --n-examples 0 \
  --score-mode forced_choice
# → babylm_finetune/logs/all_sft_evals_n0.md
```

Only full-data (`__all_types__`) BabyLM SFT runs — skips holdouts / n-scaling:

```bash
modal run -m ravens_numerical.cloud.modal_eval \
  --models babylm-sft-alltasks \
  --n-examples 0 \
  --score-mode forced_choice
# → babylm_finetune/logs/babylm_sft_alltasks_evals_n0.md
```

Eval still uses the full test set (all subtasks), regardless of train holdouts.

### Eval logs

| Kind | Path |
|------|------|
| Hub BabyLM (`BabyLM-community/...`) | `artifacts/logs/babyLMexperiments.md` |
| CHILDES ladder (base) | `artifacts/logs/childesExperiments.md` |
| TinyDialogues (base) | `artifacts/logs/tdExperiments.md` |
| SFT Volume checkpoint | `babylm_finetune/logs/<run_id>__n{N}.md` |
| BabyLM SFT only (batch) | `babylm_finetune/logs/all_sft_evals_n{N}.md` |
| BabyLM all_types SFT only | `babylm_finetune/logs/babylm_sft_alltasks_evals_n{N}.md` |
| CHILDES SFT only (batch) | `babylm_finetune/logs/childes_sft_evals_n{N}.md` |
| TinyDialogues SFT only (batch) | `babylm_finetune/logs/td_sft_evals_n{N}.md` |
| N-scaling sweep (batch) | `babylm_finetune/logs/n_scaling_evals_n{N}.md` |

Each entry’s H2 header includes `(n_examples=N)`, and the settings line repeats
``--n-examples N`` (plus the ICL test filename when N is 1 or 3). One markdown
file per finetuned `run_id` × shot count; `--models babylm-sft` /
`babylm-sft-alltasks` / `childes-sft` / `td-sft` also overwrite the combined
snapshot for that N.

## N-scaling experiment (data-size sweep)

Train BabyLM **10m** and **100m** on nested all-types pilots
`N ∈ {50,100,250,500,1000}` (10 models), fixed val + full test, then plot accuracy
vs N.

```bash
# 1) Nested pilots (does not overwrite data/sft/pilots/train_n*.jsonl)
python3 babylm_finetune/scripts/write_n_scaling_pilots.py

# 2) Modal SFT (records run_ids → outputs/n_scaling_manifest.json)
bash babylm_finetune/scripts/run_n_scaling_sft.sh

# 3) Batch eval → logs/n_scaling_evals.md + outputs/n_scaling_results.json
bash babylm_finetune/scripts/eval_n_scaling.sh

# 4) Accuracy vs N plot
python3 babylm_finetune/scripts/plot_n_scaling.py
# → babylm_finetune/outputs/n_scaling_accuracy.png
```

| Path | Role |
|------|------|
| `data/sft/pilots/n_scaling/train_n{N}.jsonl` | Nested stratified train subsets |
| `outputs/n_scaling_manifest.json` | `(model_tag, N, run_id)` table |
| `logs/n_scaling_evals.md` | One file: each `run_id` → overall / by-task accuracy |
| `outputs/n_scaling_results.json` | Same scores as JSON for plotting |
| `outputs/n_scaling_accuracy.png` | N-scaling summary plot |

Run names are `n_scaling_n{N}` (example run_id:
`babylm-10m-gpt2__n_scaling_n50__20260730T...`).

## Local train (debug)

```bash
python3 babylm_finetune/scripts/train_sft.py \
  --config babylm_finetune/configs/train/sft_babylm_10m.yaml \
  --train-file babylm_finetune/data/sft/pilots/train_n10.jsonl \
  --dry-run

python3 babylm_finetune/scripts/train_sft.py \
  --config babylm_finetune/configs/train/sft_pilot.yaml \
  --train-file babylm_finetune/data/sft/pilots/train_n10.jsonl \
  --output-dir babylm_finetune/outputs
```

## Overlap check

`check_overlaps.py` asserts no shared matrices across train/val/test, no overlap
with `IN_CONTEXT_EXAMPLES`, no within-split matrix dups, expected counts
(train 1800 / val 200 / test 500), and even small/large bands for magnitude types
(challenge test items are a frozen suite and are not 25/25 band-balanced).
When `oneICL_*` / `threeICL_*` files are present, it also checks that their
per-task ICL demos are disjoint from train/val/test, from `IN_CONTEXT_EXAMPLES`,
unique within each type (and across tasks), disjoint across splits' three-shot
files, and that each task's one-shot ICL is a prefix of its three-shot ICL.

```bash
python3 babylm_finetune/scripts/build_icl_test_sets.py
python3 babylm_finetune/scripts/check_overlaps.py
```

Eval with Modal: `--n-examples 1` (oneICL) or `--n-examples 3` (threeICL);
`--n-examples 0` is zero-shot `complete.json` (see `docs/eval.md`).
