# Evaluation guide

## Backends

| Backend | CLI flag | Server |
|---------|----------|--------|
| vLLM | `--backend vllm` | `vllm serve <hf-model-id> --host 0.0.0.0 --port 8000` |
| Ollama | `--backend ollama` | `ollama serve` + `ollama pull qwen3:8b` |
| HuggingFace | `--backend hf` | In-process (MiniBERTa RoBERTa MLM) |

## Task suites

| `--task-type` | Tasks | Data | Default `--n-examples` |
|---------------|-------|------|------------------------|
| `ravens` (default) | `ravens_numerical` | `data/tasks.json` (140) | `0 3` |
| `webb` | `ravens_numerical` | `data/tasks_webb.json` (151) | `0 3` |
| `aba` | `rules` (ABA/ABB, 70 eval) | `data/tasks_aba.json` | `0 5 10 15 20` |
| `hierarchical` | `hierarchical` (70 eval) | `data/tasks_aba.json` | `0 5 10 15 20` |

Explicit `--tasks ...` still overrides the suite when you need `matrix` / `matrix_easy`.
For `--task-type aba` / `hierarchical`, only `--n-examples` values in `{0, 5, 10, 15, 20}` are allowed.

### Webb suite

`--task-type webb` uses the original Webb et al. (2023) digit-matrix corpus
(`data/legacy/matrix/all_problems.npz`), exported to Raven-compatible JSON:

- **151 canonical tasks**: up to five non-empty test items per usable rule type
  (31 types; `AND_permuted` excluded; short `prog_size2` contributes one).
- **4 options (A–D)**: each item’s original 8 choices are reduced deterministically
  (correct answer + three distractors, then shuffled).
- **Distinct ICL**: the held-out last three problems per rule type (same reduction).
- Prompt formats, scoring modes, and backends match `--task-type ravens`
  (default: completion + forced_choice; override with `--prompt-type` /
  `--score-mode`). Multi-digit cells keep
  Webb formatting such as `[5 5]`. Logic/set rules use per-item `perm_invariant`.

Regenerate with:

```bash
PYTHONPATH=src python -m ravens_numerical.generation.export_webb
```

## Quick runs

```bash
# Pythia via vLLM
vllm serve EleutherAI/pythia-70m-deduped --host 0.0.0.0 --port 8000
./scripts/run_pythia_ravens.sh

# Qwen3 via Ollama
./scripts/run_qwen3_ravens.sh

# BabyLM GPT-2 baseline
./scripts/run_babylm_ravens.sh

# MiniBERTa (HF, no vLLM)
./scripts/run_miniberta_ravens.sh
```

Or invoke the harness directly:

```bash
# Raven's numerical (default suite)
ravens-run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --task-type ravens \
  --n-examples 0

# Webb original digit matrices (same harness / formats as ravens)
ravens-run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --task-type webb \
  --n-examples 0 3

# ABA / ABB rules
ravens-run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --task-type aba \
  --n-examples 0 5 10 15 20

# Hierarchical equality
ravens-run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --task-type hierarchical \
  --n-examples 0 5 10 15 20
```

Results are written under `artifacts/runs/{model_tag}/{run_id}/ravens_numerical/` (or `rules/` / `hierarchical/` for `--task-type aba` / `hierarchical`).

## Scoring modes

| `--score-mode` | Behavior |
|----------------|----------|
| `forced_choice` (default) | Echo logprob argmax over formatted answer options (`correct`; generation match when parseable). Ravens / Webb use completion prompts. Hierarchical only: vLLM `structured_outputs.choice` during generation. |
| `free_gen` | Parse free generation (first token / string match) |
| `auto` | `forced_choice` for Pythia `@stepN` checkpoints; `free_gen` otherwise |

Available on both `ravens-run` and Modal (`modal run -m ravens_numerical.cloud.modal_eval --score-mode ...`).
Raven's / Webb's default `--prompt-type` is `completion` (override with `--prompt-type instruction`). Raven's instruction `choice_only` is unchanged when using instruction + `free_gen`.

## Modal GPU eval

```bash
pip install -e ".[modal]"
modal setup

# Smoke test (Raven's)
modal run -m ravens_numerical.cloud.modal_eval --max-tasks 10 --models EleutherAI/pythia-70m-deduped

# Webb smoke test
modal run -m ravens_numerical.cloud.modal_eval --task-type webb --max-tasks 10 --models EleutherAI/pythia-70m-deduped

# ABA / hierarchical smoke tests
modal run -m ravens_numerical.cloud.modal_eval --task-type aba --models EleutherAI/pythia-70m-deduped
modal run -m ravens_numerical.cloud.modal_eval --task-type hierarchical --models EleutherAI/pythia-70m-deduped

# Explicit free-gen scoring (e.g. full Pythia on ABA)
modal run -m ravens_numerical.cloud.modal_eval --task-type aba --models pythia --score-mode free_gen --n-examples 15

# Full scaling ladder (appends to artifacts/logs/experiments.md)
modal run -m ravens_numerical.cloud.modal_eval --models sweep --n-examples 0
```

BabyLM models log to `artifacts/logs/babyLMexperiments.md`; Pythia checkpoints to `artifacts/logs/pythiacheckpoints.md`; OLMo 2 checkpoints to `artifacts/logs/olmocheckpoints.md`; others to `artifacts/logs/experiments.md`.

```bash
# OLMo 2 (base + instruct: 1B/7B/13B → experiments.md)
# OLMo 2 (base + instruct: 1B/7B/13B → experiments.md); ``olmo`` is an alias for ``olmo2``
modal run -m ravens_numerical.cloud.modal_eval --models olmo2 --n-examples 0

# Pythia training checkpoints (70M, 160M, 1B × 8 steps → pythiacheckpoints.md).
# Baby suites: aba uses echo logprob argmax; hierarchical uses structured choice + logprob argmax.
modal run -m ravens_numerical.cloud.modal_eval --models pythia-checkpoints --n-examples 0
modal run -m ravens_numerical.cloud.modal_eval --models pythia-checkpoints --task-type aba --n-examples 15

# OLMo 2 stage-1 checkpoints (~1B/21B/42B/49B/63B; 7B also 5B/10B; 13B also 10B → olmocheckpoints.md)
modal run -m ravens_numerical.cloud.modal_eval --models olmo2-checkpoints --n-examples 0
```

## Analysis

```bash
ravens-aggregate
# → artifacts/aggregate/scaling_results.csv
# → docs/scaling_summary.md

python -m ravens_numerical.analysis.plot_scaling
python -m ravens_numerical.analysis.plot_task_scaling
python -m ravens_numerical.analysis.audit path/to/results.json
```

Plots default to `artifacts/plots/`. Override artifact root with `RAVENS_ARTIFACTS_DIR`.
