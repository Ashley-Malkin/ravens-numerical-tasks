# baby-reasoning evaluation harness (ravens tasks)

This folder integrates **[baby-reasoning](https://github.com/Ashley-Malkin/baby-reasoning)** so you can run **`tasks.json`** benchmarks through the same **Task + backend** loop used for matrix/rules tasks—with either:

- **vLLM** (OpenAI-compatible completions API), e.g. **EleutherAI Pythia**, or  
- **Ollama** (`/api/generate`), e.g. **Qwen3** tags (`qwen3:8b`, …) aligned with [`evaluate.py`](../evaluate.py).

Shared prompting and answer parsing live in [`ravens_prompts.py`](../ravens_prompts.py) and [`ravens_answer_parse.py`](../ravens_answer_parse.py). Default model lists are centralized in [`ravens_eval_models.py`](../ravens_eval_models.py).

## Layout

- **`baby-reasoning/`** — vendored copy of [Ashley-Malkin/baby-reasoning](https://github.com/Ashley-Malkin/baby-reasoning) (MIT), extended in-tree with **`ravens_numerical`** (`baby_reasoning/tasks/ravens_numerical.py`), **`OllamaBackend`** / **`VLLMBackend`** (`baby_reasoning/model.py`), and **`--backend`** CLI wiring. All changes are committed in the parent **ravens-numerical-tasks** repo (not a git submodule).

## Prerequisites

- **Python 3.11+**
- **`uv`** (recommended) or `pip install -e .` inside `baby-reasoning/`
- **GPU + CUDA** typical for vLLM
- **`PYTHONPATH`** must include the **ravens-numerical-tasks repo root** when running `script/run`

## Backends and servers

| Backend | CLI | Server | Default URL |
|---------|-----|--------|----------------|
| **vLLM** | `--backend vllm` | `vllm serve <hf-model-id> …` | `http://localhost:8000` (`--base-url`) |
| **Ollama** | `--backend ollama` | `ollama serve` + `ollama pull qwen3:8b` … | `http://localhost:11434` (`--ollama-base-url`) |

## Quick runs from repo root

After `export PYTHONPATH="$(pwd)"` (or use the wrappers below, which set it):

### Pythia (vLLM)

```bash
vllm serve EleutherAI/pythia-70m-deduped --host 0.0.0.0 --port 8000
./run_pythia_ravens.sh
```

### Qwen3 (Ollama), same tags as `evaluate.DEBUG_MODELS`

```bash
./run_qwen3_ravens.sh
```

### Both in sequence (full suite)

```bash
./run_ravens_vllm_and_ollama.sh
```

Use **`SKIP_VLLM=1`** or **`SKIP_OLLAMA=1`** to run only one side. Override **`MODELS`** / **`PYTHIA_MODEL`** / URLs via env (see scripts).

## Manual `script/run` examples

From **`baby_reasoning_eval/baby-reasoning`**:

```bash
export RAVENS_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${RAVENS_ROOT}:${PYTHONPATH}"
uv sync

uv run script/run \
  --backend vllm \
  --models EleutherAI/pythia-70m-deduped \
  --tasks ravens_numerical \
  --ravens-repo-root "$RAVENS_ROOT" \
  --ravens-tasks-json "$RAVENS_ROOT/tasks.json" \
  --n-examples 0 \
  --base-url http://localhost:8000
```

```bash
uv run script/run \
  --backend ollama \
  --models qwen3:30b qwen3:8b \
  --tasks ravens_numerical \
  --ravens-repo-root "$RAVENS_ROOT" \
  --ravens-tasks-json "$RAVENS_ROOT/tasks.json" \
  --n-examples 0 \
  --ollama-base-url http://localhost:11434
```

### Useful flags

| Flag | Purpose |
|------|---------|
| `--ravens-max-tasks N` | Only first N problems (smoke tests). |
| `--ravens-repo-root` | Ravens repo root (contains `ravens_prompts.py`). |
| `--ravens-tasks-json` | Path to JSON (default `<ravens-repo-root>/tasks.json`). |
| `--ollama-timeout` | Seconds per Ollama request (default 300). |
| `--ollama-max-tokens` | `num_predict` for Ollama (default 64). |
| `--n-examples` | In-context demos prepended from `ravens_prompts.IN_CONTEXT_EXAMPLES` (`0` = none, `1` = one, `3` = all three). |
| `--ravens-prompt-type` | `instruction` (letter MCQ; default) or `completion` (bracket fill-in, benpry/baby-reasoning style). |

Results are written under **`baby-reasoning/results/`**. Completion runs use `{n}_examples_completion.json`; instruction runs use `{n}_examples.json`.

## Prompting and scoring

- **`instruction`** (default): prompts match **`evaluate.py` plain mode** (letter **A–D**); scoring uses **`ravens_answer_parse.parse_answer`**.
- **`completion`**: bracketed 3×3 grid ending in `[` (same format as benpry **`matrix_easy`**); model fills the blank cell; same 120 tasks and answer options; scoring compares text before `]` to the correct option value (`combine` / `intersection` are set-wise).

## Modal (Option A: vLLM inside GPU container)

[`modal_eval.py`](modal_eval.py) runs **`ravens_numerical`** on Modal with **`vllm serve`** started in the same container, then **`script/run --backend vllm`**. Use this for **Pythia vs Qwen3 scaling** on CUDA without local vLLM on macOS.

One-time:

```bash
pip install modal
modal setup
```

From the **ravens-numerical-tasks repo root**:

### `--models` (single flag for one or many HF ids)

| Value | Meaning |
|-------|---------|
| `EleutherAI/pythia-70m-deduped` | One model |
| `id1,id2,...` | Comma-separated list, run each sequentially |
| `sweep` | Full scaling ladder (8 Pythia + 5 Qwen3 from [`ravens_eval_models.py`](../ravens_eval_models.py)) |
| `pythia` | Pythia subset only |
| `qwen3` | Qwen3 subset only |

```bash
# Smoke test (10 tasks, one small model)
modal run baby_reasoning_eval/modal_eval.py \
  --max-tasks 10 --models EleutherAI/pythia-70m-deduped --n-examples 0

# Full scaling ladder (120 tasks; logs each model to experiments.md)
modal run baby_reasoning_eval/modal_eval.py --models sweep --n-examples 0

# Same ladder with completion-style prompts (benpry / matrix_easy format)
modal run baby_reasoning_eval/modal_eval.py --models sweep --n-examples 1 --prompt-type completion

# Instruction + completion per model (one vLLM load each; two experiment log entries)
modal run baby_reasoning_eval/modal_eval.py --models sweep --n-examples 1 --prompt-type both

# Instruction vs completion on one model
modal run baby_reasoning_eval/modal_eval.py \
  --models EleutherAI/pythia-410m-deduped --n-examples 1 --prompt-type instruction
modal run baby_reasoning_eval/modal_eval.py \
  --models EleutherAI/pythia-410m-deduped --n-examples 1 --prompt-type completion

# Two specific sizes
modal run baby_reasoning_eval/modal_eval.py \
  --models EleutherAI/pythia-410m-deduped,Qwen/Qwen3-4B --n-examples 0

# Optional: ICL on full ladder
modal run baby_reasoning_eval/modal_eval.py --models sweep --n-examples 3
```

Use `--n-examples 0` (zero-shot), `1` (one ICL demo), or `3` (all three demos).

| Flag | Values |
|------|--------|
| `--prompt-type` | `instruction` (default), `completion`, or `both` (instruction then completion per model, one vLLM load) |

Successful runs via the **entrypoint** (without `::`) append accuracy to [`experiments.md`](experiments.md).

GPU tier is chosen per model (`T4` for ≤1.4B, `A10G` for mid-size, `A100` for ≥10B such as Pythia-12B and Qwen3-14B); see `MODEL_GPU_TIER` in [`ravens_eval_models.py`](../ravens_eval_models.py). HF weights are cached on Modal volume **`ravens-hf-cache`**. Decoding uses **`temperature=0`** via vLLM.

Legacy direct remote functions (no `experiments.md` update):

```bash
modal run baby_reasoning_eval/modal_eval.py::run_compare_ravens --max-tasks 10
modal run baby_reasoning_eval/modal_eval.py::run_pythia_ravens --max-tasks 10
modal run baby_reasoning_eval/modal_eval.py::run_qwen3_ravens --max-tasks 10
```

### Aggregate scaling results

After local or Modal runs, JSON lives under **`baby-reasoning/results/`**. Combine into CSV + markdown:

```bash
python baby_reasoning_eval/aggregate_results.py
# → baby_reasoning_eval/scaling_results.csv
# → baby_reasoning_eval/scaling_results.md
```

### Evaluation protocol (scaling)

| Setting | Recommended |
|---------|-------------|
| Tasks | All 120 (omit `--max-tasks` for real numbers) |
| `n_examples` | `0` primary; `3` optional for ICL |
| Backend | Modal + vLLM + HF ids (both families) |
| Chance baseline | 25% (4-way MCQ; computed in aggregate output) |
