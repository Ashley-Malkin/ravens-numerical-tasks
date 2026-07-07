# Evaluation guide

## Backends

| Backend | CLI flag | Server |
|---------|----------|--------|
| vLLM | `--backend vllm` | `vllm serve <hf-model-id> --host 0.0.0.0 --port 8000` |
| Ollama | `--backend ollama` | `ollama serve` + `ollama pull qwen3:8b` |
| HuggingFace | `--backend hf` | In-process (MiniBERTa RoBERTa MLM) |

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
ravens-run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --tasks ravens_numerical \
  --n-examples 0
```

Results are written under `artifacts/runs/{model_tag}/{run_id}/ravens_numerical/`.

## Modal GPU eval

```bash
pip install -e ".[modal]"
modal setup

# Smoke test
modal run -m ravens_numerical.cloud.modal_eval --max-tasks 10 --models EleutherAI/pythia-70m-deduped

# Full scaling ladder (appends to artifacts/logs/experiments.md)
modal run -m ravens_numerical.cloud.modal_eval --models sweep --n-examples 0
```

BabyLM models log to `artifacts/logs/babyLMexperiments.md`; others to `artifacts/logs/experiments.md`.

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
