#!/usr/bin/env python3
"""Run ravens_numerical eval on Modal (Option A): vLLM server inside the GPU container.

Compare Pythia vs Qwen3 (or a size ladder) on the same ``tasks.json`` / prompts as
``baby_reasoning_eval`` locally, using ``script/run --backend vllm`` against ``http://127.0.0.1:8000``.

Setup (once on your machine):

    pip install modal
    modal setup

Smoke test:

    modal run baby_reasoning_eval/modal_eval.py --max-tasks 10 --models EleutherAI/pythia-70m-deduped

Full scaling ladder (350 tasks; appends to ``experiments.md``; BabyLM → ``babyLMexperiments.md``;
CHILDES ladder → ``childesExperiments.md``; TinyDialogues → ``tdExperiments.md``;
OpenSubtitles → ``opensubtitlesExperiments.md``; Wiki → ``wikiExperiments.md``;
TinyStories → ``tinystoriesExperiments.md``;
Pythia checkpoints → ``pythiacheckpoints.md``;
OLMo 2 checkpoints → ``olmocheckpoints.md``; saves per-trial JSON under ``artifacts/runs/``):

    modal run -m ravens_numerical.cloud.modal_eval --models sweep --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models childes --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models td --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models os --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models wiki --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models ts --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models pythia-checkpoints --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models olmo2-checkpoints --n-examples 0

BabyLM SFT Volume checkpoint (``ravens-babylm-sft`` / ``/checkpoints/<run_id>``;
logs → ``babylm_finetune/logs/<run_id>__n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models babylm-10m-gpt2__all_types__20260723T212815Z \\
      --n-examples 0 --score-mode forced_choice

ICL-augmented Raven test sets (same 500 complete tasks + disjoint ICL bank):
``--n-examples 1`` → ``oneICL_test.json``; ``--n-examples 3`` → ``threeICL_test.json``;
``--n-examples 0`` → zero-shot ``complete.json``:

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models <run_id> \\
      --n-examples 1 --score-mode forced_choice

5-digit OOD suite (same structure, ``n → 10000+n``; alias ``5digit``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models miniberta-sft --n-examples 0 --score-mode forced_choice \\
      --ravens-tasks-json 5digit

Challenge suite (3 hard types × 50; alias ``challenge``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models miniberta-sft --n-examples 0 --score-mode forced_choice \\
      --ravens-tasks-json challenge

BabyLM SFT only (``run_id`` starts with ``babylm-`` →
``babylm_finetune/logs/all_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models babylm-sft --n-examples 0 --score-mode forced_choice

BabyLM full-data SFT only (``__all_types__``; skips holdouts / n-scaling →
``babylm_finetune/logs/babylm_sft_alltasks_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models babylm-sft-alltasks --n-examples 0 --score-mode forced_choice

CHILDES ladder SFT only (``run_id`` starts with ``childes-`` →
``babylm_finetune/logs/childes_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models childes-sft --n-examples 0 --score-mode forced_choice

TinyDialogues SFT only (``run_id`` starts with ``td-`` →
``babylm_finetune/logs/td_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models td-sft --n-examples 0 --score-mode forced_choice

OpenSubtitles SFT only (``run_id`` starts with ``os-`` →
``babylm_finetune/logs/os_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models os-sft --n-examples 0 --score-mode forced_choice

Wiki SFT only (``run_id`` starts with ``wiki-`` →
``babylm_finetune/logs/wiki_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models wiki-sft --n-examples 0 --score-mode forced_choice

TinyStories SFT only (``run_id`` starts with ``ts-`` →
``babylm_finetune/logs/ts_sft_evals_n{N}.md``):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models ts-sft --n-examples 0 --score-mode forced_choice

MiniBERTa MLM SFT only (``run_id`` starts with ``miniberta-`` →
``babylm_finetune/logs/miniberta_sft_evals_n{N}.md``; HF PLL, not vLLM):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models miniberta-sft --n-examples 0 --score-mode forced_choice

N-scaling subset (comma-separated run_ids → dedicated combined log):

    modal run -m ravens_numerical.cloud.modal_eval \\
      --models id1,id2 --n-examples 0 --score-mode forced_choice \\
      --combined-sft-log babylm_finetune/logs/n_scaling_evals.md \\
      --combined-sft-results-json babylm_finetune/outputs/n_scaling_results.json

Direct remote functions (``::run_ravens_eval_t4``, etc.) skip experiment logging; use the entrypoint above.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, TextIO

import modal

# Paths on the local machine (repo layout).
_MODAL_EVAL_DIR = Path(__file__).resolve().parent
from ravens_numerical.paths import (  # noqa: E402
    BABYLM_EXPERIMENTS_MD,
    BABYLM_FINETUNE_LOGS_DIR,
    BABYLM_SFT_ALL_EVALS_MD,
    BABYLM_SFT_ALLTASKS_EVALS_MD,
    CHILDES_EXPERIMENTS_MD,
    CHILDES_SFT_ALL_EVALS_MD,
    CONTAINER_RAVENS_ROOT,
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    OS_EXPERIMENTS_MD,
    OS_SFT_ALL_EVALS_MD,
    PYTHIA_CHECKPOINTS_MD,
    REPO_ROOT,
    RUNS_DIR,
    TD_EXPERIMENTS_MD,
    TD_SFT_ALL_EVALS_MD,
    TS_EXPERIMENTS_MD,
    TS_SFT_ALL_EVALS_MD,
    WIKI_EXPERIMENTS_MD,
    WIKI_SFT_ALL_EVALS_MD,
    MINIBERTA_SFT_ALL_EVALS_MD,
    babylm_sft_experiments_md,
    combined_sft_log_path_for_n_examples,
)


def _parse_n_scaling_n(run_id: str) -> Optional[int]:
    """Extract N from ``...__n_scaling_n{N}__...`` run ids; else None."""
    for part in run_id.split("__"):
        if part.startswith("n_scaling_n"):
            suffix = part[len("n_scaling_n") :]
            if suffix.isdigit():
                return int(suffix)
    return None


def _write_combined_sft_results_json(
    path: Path,
    *,
    results_by_run_id: dict[str, dict[str, Any]],
    n_examples: int | None = None,
) -> Path:
    """Write plot-friendly JSON rows labeled by ``run_id`` (and N when present)."""
    rows: list[dict[str, Any]] = []
    for run_id in sorted(results_by_run_id):
        summary = results_by_run_id[run_id]
        model_tag = run_id.split("__", 1)[0]
        row: dict[str, Any] = {
            "run_id": run_id,
            "model_tag": model_tag,
            "accuracy": summary.get("accuracy"),
            "correct": summary.get("correct"),
            "total": summary.get("total"),
            "by_task_type": summary.get("by_task_type", {}),
        }
        if n_examples is not None:
            row["n_examples"] = n_examples
        n_val = _parse_n_scaling_n(run_id)
        if n_val is not None:
            row["N"] = n_val
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"results": rows}
    if n_examples is not None:
        payload["n_examples"] = n_examples
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

# Paths inside the Modal container (repo copied here via ``add_local_dir``).
CONTAINER_TASKS_JSON = f"{CONTAINER_RAVENS_ROOT}/data/complete.json"
CONTAINER_TASKS_5DIGIT_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks_5digit.json"
CONTAINER_CHALLENGE_TASKS_JSON = f"{CONTAINER_RAVENS_ROOT}/data/challenge_tasks.json"
CONTAINER_TASKS_LEGACY_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks.json"
CONTAINER_TASKS_ABA_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks_aba.json"
CONTAINER_TASKS_WEBB_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks_webb.json"
CONTAINER_BABYLM_DATA = f"{CONTAINER_RAVENS_ROOT}/babylm_finetune/data"
CONTAINER_RUNS_DIR = f"{CONTAINER_RAVENS_ROOT}/artifacts/runs"
CONTAINER_SRC = f"{CONTAINER_RAVENS_ROOT}/src"

# Raven ``--n-examples`` → disjoint ICL test JSON (same tasks as test.json).
_RAVENS_ICL_TEST_BY_N: dict[int, str] = {
    1: "oneICL_test.json",
    3: "threeICL_test.json",
}

# Shorthand ``--ravens-tasks-json`` values → repo-relative paths.
_RAVENS_TASKS_JSON_ALIASES: dict[str, str] = {
    "default": "data/complete.json",
    "complete": "data/complete.json",
    "complete.json": "data/complete.json",
    "tasks": "data/tasks.json",
    "tasks.json": "data/tasks.json",
    "5digit": "data/tasks_5digit.json",
    "five_digit": "data/tasks_5digit.json",
    "tasks_5digit": "data/tasks_5digit.json",
    "tasks_5digit.json": "data/tasks_5digit.json",
    "challenge": "data/challenge_tasks.json",
    "challenge_tasks": "data/challenge_tasks.json",
    "challenge_tasks.json": "data/challenge_tasks.json",
}


def resolve_ravens_tasks_json_arg(spec: str) -> Path:
    """Resolve a ``--ravens-tasks-json`` alias or path under the repo."""
    raw = spec.strip()
    if not raw:
        raise ValueError("--ravens-tasks-json must be a non-empty path or alias")
    key = raw.lower().replace("-", "_")
    if key in _RAVENS_TASKS_JSON_ALIASES:
        return (REPO_ROOT / _RAVENS_TASKS_JSON_ALIASES[key]).resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Raven tasks JSON not found: {spec!r} (resolved {path}). "
            "Use an existing path or alias: "
            + ", ".join(sorted(set(_RAVENS_TASKS_JSON_ALIASES)))
        )
    return path


def resolve_container_ravens_tasks_json(
    *,
    task_type: str,
    n_examples: int,
    ravens_tasks_json: str | None,
) -> str | None:
    """Container path for Raven tasks JSON, or None for the suite default.

    Explicit ``--ravens-tasks-json`` wins over ICL ``--n-examples`` file selection.
    """
    if ravens_tasks_json:
        host = resolve_ravens_tasks_json_arg(ravens_tasks_json)
        if n_examples in _RAVENS_ICL_TEST_BY_N and host.name in {
            "tasks_5digit.json",
            "challenge_tasks.json",
        }:
            label = (
                "5-digit"
                if host.name == "tasks_5digit.json"
                else "challenge"
            )
            print(
                f"Note: --ravens-tasks-json {ravens_tasks_json!r} with "
                f"--n-examples {n_examples}: using the {label} zero-shot file "
                f"(no {label} ICL test set); prompt ICL bank may still use "
                "default-suite demos.",
                flush=True,
            )
        return container_path_for_repo_file(host)
    return ravens_tasks_json_for_n_examples(n_examples, task_type)


def resolve_icl_test_data_path(filename: str) -> Path:
    """Resolve an ICL test JSON under ``babylm_finetune/data/`` or ``data/``."""
    name = Path(filename).name
    candidates = [
        REPO_ROOT / "babylm_finetune" / "data" / name,
        REPO_ROOT / "data" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"ICL test set {name!r} not found; looked in babylm_finetune/data/ and data/. "
        "Run babylm_finetune/scripts/build_icl_test_sets.py"
    )


def container_path_for_repo_file(path: Path) -> str:
    """Map a repo-relative host path to the Modal container path."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"test data must live under the repo root ({REPO_ROOT}); got {path}"
        ) from exc
    return f"{CONTAINER_RAVENS_ROOT}/{rel.as_posix()}"


def ravens_tasks_json_for_n_examples(
    n_examples: int,
    task_type: str,
) -> str | None:
    """Container tasks JSON for Raven ICL test sets, else None (suite default).

    ``--n-examples 1`` → ``oneICL_test.json``; ``--n-examples 3`` →
    ``threeICL_test.json``. Zero-shot and other counts use the suite default
    (``tasks.json`` / ``tasks_webb.json``) with the usual prompt ICL bank.
    """
    if task_type != "ravens":
        return None
    filename = _RAVENS_ICL_TEST_BY_N.get(n_examples)
    if filename is None:
        return None
    return container_path_for_repo_file(resolve_icl_test_data_path(filename))

EASY_TASK_DIRS = {
    "aba": ("rules",),
    "hierarchical": ("hierarchical",),
}
BABY_TASK_TYPES = frozenset(EASY_TASK_DIRS)
RAVENS_LIKE_TASK_TYPES = frozenset({"ravens", "webb"})
SUPPORTED_TASK_TYPES = frozenset({"ravens", "webb", *BABY_TASK_TYPES})

EXPERIMENTS_MD_LOCAL = EXPERIMENTS_MD
BABYLM_EXPERIMENTS_MD_LOCAL = BABYLM_EXPERIMENTS_MD
CHILDES_EXPERIMENTS_MD_LOCAL = CHILDES_EXPERIMENTS_MD
TD_EXPERIMENTS_MD_LOCAL = TD_EXPERIMENTS_MD
OS_EXPERIMENTS_MD_LOCAL = OS_EXPERIMENTS_MD
WIKI_EXPERIMENTS_MD_LOCAL = WIKI_EXPERIMENTS_MD
TS_EXPERIMENTS_MD_LOCAL = TS_EXPERIMENTS_MD
PYTHIA_CHECKPOINTS_MD_LOCAL = PYTHIA_CHECKPOINTS_MD
OLMO_CHECKPOINTS_MD_LOCAL = OLMO_CHECKPOINTS_MD
LOCAL_RUNS_DIR = RUNS_DIR

BABYLM_EXPERIMENTS_PREAMBLE = (
    "# BabyLM Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models babylm` "
    "and other BabyLM HuggingFace ids.\n\n"
)

CHILDES_EXPERIMENTS_PREAMBLE = (
    "# CHILDES GPT-2 ladder Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models childes` "
    "(mcxfrank/childes-gpt2-ladder budgets 1M / 5M / 12M / 24M at seed42).\n\n"
)

TD_EXPERIMENTS_PREAMBLE = (
    "# TinyDialogues GPT-2 Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models td` "
    "(GPT2-small_TD_{10,20,50,100,200}M_20-epochs_seed42 on Volume "
    "`ravens-td-base`).\n\n"
)

OS_EXPERIMENTS_PREAMBLE = (
    "# OpenSubtitles GPT-2 Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models os` "
    "(GPT2-small_opensubtitles_20M seeds 0/42/123 on Volume `ravens-os-base`).\n\n"
)

WIKI_EXPERIMENTS_PREAMBLE = (
    "# Wiki GPT-2 Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models wiki` "
    "(GPT2-small_wiki_20M seeds 0/42/123 on Volume `ravens-wiki-base`).\n\n"
)

TS_EXPERIMENTS_PREAMBLE = (
    "# TinyStories GPT-2 Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models ts` "
    "(GPT2-small_tinystories_10m_1e-04 on Volume `ravens-ts-base`).\n\n"
)

PYTHIA_CHECKPOINTS_PREAMBLE = (
    "# Pythia checkpoint Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models pythia-checkpoints` "
    "and other Pythia ids with `@stepN` HuggingFace revisions.\n\n"
)

OLMO_CHECKPOINTS_PREAMBLE = (
    "# OLMo 2 checkpoint Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models olmo2-checkpoints` "
    "and other OLMo 2 ids with `@stage1-...` HuggingFace revisions "
    "(1B / 21B / 42B / 49B tokens-viewed budgets; nearest published branch).\n\n"
)

from ravens_numerical.cloud.modal_results_export import (
    attach_results_data,
    export_run_results,
    public_summary,
)
from ravens_numerical.models.registry import (
    DEFAULT_BABYLM_VLLM,
    DEFAULT_MINIBERTA_VLLM,
    DEFAULT_PYTHIA_VLLM,
    DEFAULT_QWEN3_VLLM,
    base_model_id,
    checkpoint_step_from_model_id,
    gpu_tier_for_model,
    is_babylm_model,
    is_childes_model,
    is_miniberta_model,
    is_miniberta_sft_checkpoint,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    is_qwen3_instruct_model,
    is_qwen3_model,
    is_sft_checkpoint_model,
    is_sft_run_id,
    is_td_model,
    is_os_model,
    is_wiki_model,
    is_ts_model,
    max_model_len_for_model,
    normalize_childes_model_id,
    normalize_os_model_id,
    normalize_td_model_id,
    normalize_wiki_model_id,
    normalize_ts_model_id,
    parse_checkpoint_model_id,
    resolve_childes_local_path,
    resolve_instruction_prompt_mode,
    resolve_models_arg,
    resolve_os_local_path,
    resolve_sft_model_id,
    resolve_td_local_path,
    resolve_wiki_local_path,
    resolve_ts_local_path,
    sft_run_id_from_model_id,
)

def _experiment_settings_note(model_id: str, score_mode: str | None = None) -> str:
    hf_model_id, revision = parse_checkpoint_model_id(model_id)
    if is_miniberta_model(hf_model_id) or is_miniberta_sft_checkpoint(model_id):
        max_len = max_model_len_for_model(model_id)
        note = (
            f"Modal HF transformers (`modal_eval.py`); PLL scoring; "
            f"max_len={max_len}; T4"
        )
    else:
        max_len = max_model_len_for_model(hf_model_id)
        note = (
            f"Modal vLLM (`modal_eval.py`); `--max-model-len {max_len}`; "
            "T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`"
        )
    if revision:
        note += f"; HF revision `{revision}`"
    resolved = score_mode if score_mode is not None else _score_mode_for_model(model_id)
    if resolved == "forced_choice":
        note += "; `--score-mode forced_choice` (echo logprob argmax; hierarchical also structured choice)"
    elif resolved == "free_gen":
        note += "; `--score-mode free_gen`"
    return note


SCORE_MODES = ("auto", "free_gen", "forced_choice")


def _score_mode_for_model(model_id: str, score_mode: str = "forced_choice") -> str:
    """Resolve Modal ``--score-mode`` for a model id.

    Default and explicit ``forced_choice`` win. ``free_gen`` disables it.
    ``auto`` → ``forced_choice`` for Pythia ``@stepN`` checkpoints, else ``auto``
    (CLI expands the rest to ``free_gen``).
    """
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}"
        )
    if score_mode == "forced_choice":
        return "forced_choice"
    if score_mode == "free_gen":
        return "free_gen"
    if is_pythia_checkpoint_model_id(model_id):
        return "forced_choice"
    return "auto"

DEFAULT_N_EXAMPLES = 0
PROMPT_TYPES = ("instruction", "completion")
PROMPT_TYPE_CLI = PROMPT_TYPES + ("both",)


def resolve_prompt_types(prompt_type: str) -> list[str]:
    """Expand CLI ``both`` to concrete eval prompt types."""
    if prompt_type == "both":
        return list(PROMPT_TYPES)
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(
            f"prompt_type must be one of {PROMPT_TYPE_CLI}, got {prompt_type!r}"
        )
    return [prompt_type]


def _results_filename(
    n_examples: int,
    prompt_type: str,
    prompt_mode: str | None = None,
    score_mode: str | None = None,
) -> str:
    if prompt_type == "completion":
        return f"{n_examples}_examples_completion.json"
    if prompt_mode == "choice_only":
        return f"{n_examples}_examples_choice_only.json"
    if score_mode == "forced_choice":
        return f"{n_examples}_examples_forced_choice.json"
    return f"{n_examples}_examples.json"

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

IGNORE_COPY = [
    "**/.venv/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/artifacts/**",
    "**/*.egg-info/**",
    "**/node_modules/**",
    "**/.pytest_cache/**",
    "**/open-subtitles-models/**",
    "**/wiki-models/**",
    "**/tiny-stories-model/**",
]

app = modal.App("ravens-baby-reasoning-vllm")

hf_cache_volume = modal.Volume.from_name("ravens-hf-cache", create_if_missing=True)
sft_volume = modal.Volume.from_name("ravens-babylm-sft", create_if_missing=True)
td_base_volume = modal.Volume.from_name("ravens-td-base", create_if_missing=True)
os_base_volume = modal.Volume.from_name("ravens-os-base", create_if_missing=True)
wiki_base_volume = modal.Volume.from_name("ravens-wiki-base", create_if_missing=True)
ts_base_volume = modal.Volume.from_name("ravens-ts-base", create_if_missing=True)

# HF Hub cache + BabyLM SFT checkpoints + TD / OS / Wiki / TS base weights.
EVAL_VOLUMES = {
    "/root/.cache/huggingface": hf_cache_volume,
    "/checkpoints": sft_volume,
    "/td-base": td_base_volume,
    "/os-base": os_base_volume,
    "/wiki-base": wiki_base_volume,
    "/ts-base": ts_base_volume,
}

# Lightweight image for Volume listing (no vLLM).
_list_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests>=2.31")
)


@app.function(
    image=_list_image,
    volumes={"/checkpoints": sft_volume},
    timeout=120,
)
def list_sft_checkpoint_run_ids() -> list[str]:
    """Return sorted ``run_id`` directory names under ``/checkpoints``."""
    sft_volume.reload()
    root = Path("/checkpoints")
    if not root.is_dir():
        return []
    run_ids: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        # SFT run_ids look like ``babylm-10m-gpt2__all_types__20260723T212815Z``.
        if "__" not in path.name:
            continue
        run_ids.append(path.name)
    return run_ids


def discover_local_sft_run_ids(*, prefix: str | None = None) -> list[str]:
    """Fallback: unique run_ids from local SFT log filenames / run export dirs.

    When ``prefix`` is set (e.g. ``childes-``), only matching run_ids are kept.
    """
    found: set[str] = set()
    if BABYLM_FINETUNE_LOGS_DIR.is_dir():
        for path in BABYLM_FINETUNE_LOGS_DIR.glob("*.md"):
            if path.name == BABYLM_SFT_ALL_EVALS_MD.name:
                continue
            if path.name == BABYLM_SFT_ALLTASKS_EVALS_MD.name:
                continue
            if path.name == CHILDES_SFT_ALL_EVALS_MD.name:
                continue
            if path.name == TD_SFT_ALL_EVALS_MD.name:
                continue
            stem = _sft_run_id_from_log_stem(path.stem)
            if is_sft_run_id(stem):
                found.add(stem)
    if LOCAL_RUNS_DIR.is_dir():
        for path in LOCAL_RUNS_DIR.iterdir():
            if not path.is_dir():
                continue
            name = path.name
            if name.startswith("--checkpoints--"):
                run_id = name[len("--checkpoints--") :]
                if is_sft_run_id(run_id):
                    found.add(run_id)
    run_ids = sorted(found)
    if prefix:
        run_ids = [rid for rid in run_ids if rid.startswith(prefix)]
    return run_ids


def resolve_sft_models_arg(
    *,
    prefix: str | None = None,
    contains: str | None = None,
) -> list[str]:
    """List Volume SFT checkpoints; fall back to local log/run discovery.

    ``prefix`` filters run_ids (e.g. ``childes-`` for CHILDES ladder SFT only).
    ``contains`` further requires a substring (e.g. ``__all_types__``).
    """
    try:
        run_ids = list_sft_checkpoint_run_ids.remote()
    except Exception as exc:  # noqa: BLE001 — Modal/network may be unavailable
        print(
            f"Warning: could not list Volume checkpoints ({exc}); "
            "falling back to local run_ids",
            flush=True,
        )
        run_ids = []
    if prefix:
        run_ids = [rid for rid in run_ids if rid.startswith(prefix)]
    if contains:
        run_ids = [rid for rid in run_ids if contains in rid]
    if not run_ids:
        run_ids = discover_local_sft_run_ids(prefix=prefix)
        if contains:
            run_ids = [rid for rid in run_ids if contains in rid]
        if run_ids:
            print(
                f"Using {len(run_ids)} local SFT run_id(s) "
                f"(Volume list empty or unavailable)",
                flush=True,
            )
    if not run_ids:
        label_parts: list[str] = []
        if prefix:
            label_parts.append(f"prefix {prefix!r}")
        if contains:
            label_parts.append(f"containing {contains!r}")
        label = (" matching " + " and ".join(label_parts)) if label_parts else ""
        raise ValueError(
            f"No SFT checkpoints{label} found on Volume 'ravens-babylm-sft' "
            f"or under {BABYLM_FINETUNE_LOGS_DIR}."
        )
    return [resolve_sft_model_id(rid) for rid in run_ids]


def resolve_babylm_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``babylm-``."""
    return resolve_sft_models_arg(prefix="babylm-")


def resolve_babylm_sft_alltasks_models_arg() -> list[str]:
    """BabyLM SFT checkpoints trained on full data (``__all_types__`` only).

    Excludes holdouts (``only_*`` / ``*_only_*``), n-scaling, and ICL-adjusted runs.
    """
    return resolve_sft_models_arg(prefix="babylm-", contains="__all_types__")


def resolve_childes_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``childes-``."""
    return resolve_sft_models_arg(prefix="childes-")


def resolve_td_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``td-``."""
    return resolve_sft_models_arg(prefix="td-")


def resolve_os_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``os-``."""
    return resolve_sft_models_arg(prefix="os-")


def resolve_wiki_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``wiki-``."""
    return resolve_sft_models_arg(prefix="wiki-")


def resolve_ts_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``ts-``."""
    return resolve_sft_models_arg(prefix="ts-")


def resolve_miniberta_sft_models_arg() -> list[str]:
    """List Volume SFT checkpoints whose run_id starts with ``miniberta-``."""
    return resolve_sft_models_arg(prefix="miniberta-")


eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.18.0",
        "transformers>=4.40",
        "accelerate>=0.30",
        "requests>=2.31",
        "tyro>=0.9",
        "pandas>=2.3",
    )
    .env({"PYTHONPATH": CONTAINER_SRC})
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=CONTAINER_RAVENS_ROOT,
        copy=True,
        ignore=IGNORE_COPY,
    )
)


def _vllm_log_path(model_id: str) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return Path(f"/tmp/vllm_{safe}.log")


def _vllm_log_tail(log_path: Path, max_lines: int = 200) -> str:
    if not log_path.is_file():
        return "(no vLLM log file)"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "(vLLM log empty)"
    return "\n".join(lines[-max_lines:])


def _stream_subprocess_output(stream, log_file: TextIO) -> None:
    """Read vLLM stdout/stderr line-by-line; write to log file and Modal logs."""
    try:
        for line in iter(stream.readline, ""):
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
    finally:
        stream.close()


def _wait_for_vllm(
    proc: subprocess.Popen,
    log_path: Path,
    serve_cmd: list[str],
    timeout_sec: int = 900,
) -> None:
    """Poll vLLM until the OpenAI models endpoint responds."""
    url = f"{VLLM_BASE_URL.rstrip('/')}/v1/models"
    deadline = time.time() + timeout_sec
    last_err: Optional[str] = None
    cmd_str = " ".join(serve_cmd)

    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"vLLM exited early with code {rc}.\n"
                f"Command: {cmd_str}\n"
                f"Log: {log_path}\n"
                f"Last log lines:\n{_vllm_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)
        time.sleep(2)

    raise RuntimeError(
        f"vLLM did not become ready at {url} within {timeout_sec}s: {last_err}\n"
        f"Command: {cmd_str}\n"
        f"Log: {log_path}\n"
        f"Last log lines:\n{_vllm_log_tail(log_path)}"
    )


def _start_vllm_server(
    serve_cmd: list[str],
    log_path: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen, threading.Thread, TextIO]:
    """Start ``vllm serve``; stream output to Modal logs and ``log_path`` (no PIPE deadlock)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    print("Starting vLLM:", " ".join(serve_cmd), flush=True)
    print(f"vLLM log file: {log_path}", flush=True)

    proc = subprocess.Popen(
        serve_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    reader = threading.Thread(
        target=_stream_subprocess_output,
        args=(proc.stdout, log_file),
        daemon=True,
    )
    reader.start()
    return proc, reader, log_file


def _stop_vllm_server(
    proc: subprocess.Popen,
    log_file: TextIO,
    reader: threading.Thread,
) -> None:
    log_file.flush()
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    reader.join(timeout=5)
    log_file.close()


def _attention_backend_for_gpu(gpu_tier: str) -> str:
    """T4 (sm_75) cannot use FLASH_ATTN in vLLM 0.18+; use TRITON_ATTN instead."""
    if gpu_tier == "T4":
        return "TRITON_ATTN"
    return "FLASH_ATTN"


def _build_vllm_serve_cmd(
    model_id: str,
    gpu_tier: str,
    revision: str | None = None,
) -> list[str]:
    hf_model_id = base_model_id(model_id)
    max_len = max_model_len_for_model(hf_model_id)
    cmd = [
        "vllm",
        "serve",
        hf_model_id,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--attention-backend",
        _attention_backend_for_gpu(gpu_tier),
        "--max-model-len",
        str(max_len),
    ]
    if revision:
        cmd.extend(["--revision", revision])
    if is_qwen3_instruct_model(hf_model_id):
        cmd.extend(
            [
                "--reasoning-parser",
                "qwen3",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        )
    return cmd


def _collect_latest_results(
    results_root: Path,
    *,
    task_type: str,
    n_examples: int,
    prompt_type: str,
    ravens_prompt_mode: str,
    score_mode: str = "forced_choice",
) -> Path:
    """Return the results JSON to summarize."""
    if task_type in BABY_TASK_TYPES:
        if score_mode == "forced_choice":
            results_name = f"{n_examples}_examples_forced_choice.json"
        else:
            results_name = f"{n_examples}_examples.json"
        task_dirs = EASY_TASK_DIRS[task_type]
        paths: list[Path] = []
        for task_dir in task_dirs:
            candidates = sorted(
                results_root.glob(f"**/{task_dir}/{results_name}"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No results under {results_root}; expected {task_dir}/{results_name}"
                )
            paths.append(candidates[-1])
        if len(paths) == 1:
            return paths[0]
        combined: list[Any] = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Unexpected results shape in {path}")
            combined.extend(data)
        combined_path = paths[0].parent.parent / task_type / results_name
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        return combined_path

    instruction_mode = (
        resolve_instruction_prompt_mode(ravens_prompt_mode)
        if prompt_type == "instruction"
        else None
    )
    results_name = _results_filename(n_examples, prompt_type, instruction_mode)
    results_glob = f"**/ravens_numerical/{results_name}"
    candidates = sorted(
        results_root.glob(results_glob),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No results under {results_root}; expected ravens_numerical/{results_name}"
        )
    return candidates[-1]


def _build_eval_cli_cmd(
    *,
    backend: str,
    model: str,
    n_examples: int,
    task_type: str,
    prompt_type: str,
    ravens_prompt_mode: str,
    max_tasks: Optional[int],
    base_url: str | None = None,
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "ravens_numerical.eval.cli",
        "--backend",
        backend,
        "--models",
        model,
        "--task-type",
        task_type,
        "--results-dir",
        CONTAINER_RUNS_DIR,
        "--n-examples",
        str(n_examples),
        "--score-mode",
        score_mode,
    ]
    if base_url is not None:
        cmd.extend(["--base-url", base_url])
    if task_type in BABY_TASK_TYPES:
        cmd.extend(["--aba-tasks-json", CONTAINER_TASKS_ABA_JSON])
        return cmd

    if ravens_tasks_json is not None:
        tasks_json = ravens_tasks_json
    else:
        tasks_json = (
            CONTAINER_TASKS_WEBB_JSON if task_type == "webb" else CONTAINER_TASKS_JSON
        )
    cmd.extend(
        [
            "--ravens-tasks-json",
            tasks_json,
            "--ravens-prompt-type",
            prompt_type,
        ]
    )
    if prompt_type == "instruction":
        cmd.extend(["--ravens-prompt-mode", ravens_prompt_mode])
    if max_tasks is not None:
        cmd.extend(["--ravens-max-tasks", str(max_tasks)])
    return cmd


def _run_baby_reasoning_cli(
    served: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str,
    env: dict[str, str],
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
) -> Path:
    """Run baby-reasoning CLI for one prompt type (vLLM must already be up)."""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"prompt_type must be one of {PROMPT_TYPES}, got {prompt_type!r}")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"task_type must be 'ravens', 'webb', 'aba', or 'hierarchical', got {task_type!r}"
        )

    cmd = _build_eval_cli_cmd(
        backend="vllm",
        model=served,
        n_examples=n_examples,
        task_type=task_type,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        max_tasks=max_tasks,
        base_url=VLLM_BASE_URL,
        score_mode=score_mode,
        ravens_tasks_json=ravens_tasks_json,
    )

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=CONTAINER_RAVENS_ROOT, env=env)

    return _collect_latest_results(
        Path(CONTAINER_RUNS_DIR),
        task_type=task_type,
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        score_mode=score_mode,
    )


def _run_hf_baby_reasoning_cli(
    model_id: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str,
    env: dict[str, str],
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
) -> Path:
    """Run baby-reasoning CLI with in-process HuggingFace backend (MiniBERTa)."""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"prompt_type must be one of {PROMPT_TYPES}, got {prompt_type!r}")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"task_type must be 'ravens', 'webb', 'aba', or 'hierarchical', got {task_type!r}"
        )

    cmd = _build_eval_cli_cmd(
        backend="hf",
        model=model_id,
        n_examples=n_examples,
        task_type=task_type,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        max_tasks=max_tasks,
        score_mode=score_mode,
        ravens_tasks_json=ravens_tasks_json,
    )

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=CONTAINER_RAVENS_ROOT, env=env)

    return _collect_latest_results(
        Path(CONTAINER_RUNS_DIR),
        task_type=task_type,
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        score_mode=score_mode,
    )


def _run_hf_model_eval(
    model_id: str,
    max_tasks: Optional[int],
    gpu_tier: str,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
    display_model_id: str | None = None,
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """MiniBERTa eval via HuggingFace transformers (no vLLM server)."""
    log_model_id = display_model_id or model_id
    run_types = resolve_prompt_types(prompt_type)
    env = os.environ.copy()
    env["PYTHONPATH"] = CONTAINER_SRC

    if len(run_types) == 1:
        pt = run_types[0]
        results_path = _run_hf_baby_reasoning_cli(
            model_id,
            max_tasks,
            n_examples,
            pt,
            env,
            ravens_prompt_mode,
            task_type=task_type,
            score_mode=score_mode,
            ravens_tasks_json=ravens_tasks_json,
        )
        summary = _make_model_summary(
            results_path,
            model_id=log_model_id,
            prompt_type=pt,
            n_examples=n_examples,
            gpu_tier=gpu_tier,
            include_results_data=True,
            task_type=task_type,
        )
        print(json.dumps(public_summary(summary), indent=2), flush=True)
        hf_cache_volume.commit()
        return summary

    summaries: dict[str, dict[str, Any]] = {}
    for pt in run_types:
        print(f"\n--- prompt_type={pt} ---\n", flush=True)
        results_path = _run_hf_baby_reasoning_cli(
            model_id,
            max_tasks,
            n_examples,
            pt,
            env,
            ravens_prompt_mode if pt == "instruction" else "auto",
            task_type=task_type,
            score_mode=score_mode,
            ravens_tasks_json=ravens_tasks_json,
        )
        summary = _make_model_summary(
            results_path,
            model_id=log_model_id,
            prompt_type=pt,
            n_examples=n_examples,
            gpu_tier=gpu_tier,
            include_results_data=True,
            task_type=task_type,
        )
        summaries[pt] = summary
        print(json.dumps(public_summary(summary), indent=2), flush=True)
    hf_cache_volume.commit()
    return summaries


def _vllm_served_name(model_id: str, vllm_served_name: Optional[str] = None) -> str:
    """Model name vLLM exposes on ``/v1/models`` (base HF id, no ``@stepN``)."""
    if vllm_served_name:
        return vllm_served_name
    return base_model_id(model_id)


def _run_script(
    model_id: str,
    max_tasks: Optional[int],
    gpu_tier: str,
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    hf_revision: str | None = None,
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
) -> Path:
    """Start vLLM, run baby-reasoning CLI, return path to results JSON."""
    run_types = resolve_prompt_types(prompt_type)
    if len(run_types) != 1:
        raise ValueError(
            f"_run_script expects a single prompt type; got {prompt_type!r}. "
            "Use _run_model_eval for ``both``."
        )
    pt = run_types[0]
    served = _vllm_served_name(model_id, vllm_served_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = CONTAINER_SRC
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    serve_cmd = _build_vllm_serve_cmd(model_id, gpu_tier, revision=hf_revision)
    log_path = _vllm_log_path(model_id)
    proc, reader, log_file = _start_vllm_server(serve_cmd, log_path, env)
    try:
        _wait_for_vllm(proc, log_path, serve_cmd)
        return _run_baby_reasoning_cli(
            served,
            max_tasks,
            n_examples,
            pt,
            env,
            ravens_prompt_mode,
            task_type=task_type,
            score_mode=score_mode,
            ravens_tasks_json=ravens_tasks_json,
        )
    finally:
        _stop_vllm_server(proc, log_file, reader)


def _stimulus_subtype(row: dict[str, Any]) -> str:
    """Label for per-type accuracy: Raven task_type, ABA rule, or hierarchical pattern."""
    meta = row.get("stimulus", {}).get("metadata", {}) or {}
    task = meta.get("task", {})
    if isinstance(task, dict) and task.get("task_type") is not None:
        return str(task["task_type"])
    if meta.get("rule") is not None:
        return str(meta["rule"])
    if meta.get("pattern") is not None:
        return str(meta["pattern"])
    return "unknown"


def _summarize_results(results_path: Path) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected results shape in {results_path}")

    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in data:
        correct = bool(row.get("score", {}).get("correct", False))
        by_type[_stimulus_subtype(row)].append(correct)

    total = len(data)
    n_correct = sum(1 for row in data if row.get("score", {}).get("correct"))
    return {
        "results_path": str(results_path),
        "total": total,
        "correct": n_correct,
        "accuracy": n_correct / total if total else 0.0,
        "by_task_type": {
            t: {
                "correct": sum(v),
                "total": len(v),
                "accuracy": sum(v) / len(v) if v else 0.0,
            }
            for t, v in sorted(by_type.items())
        },
    }


def _make_model_summary(
    results_path: Path,
    *,
    model_id: str,
    prompt_type: str,
    n_examples: int,
    gpu_tier: str,
    vllm_served_name: Optional[str] = None,
    include_results_data: bool = False,
    hf_revision: str | None = None,
    task_type: str = "ravens",
) -> dict[str, Any]:
    summary = _summarize_results(results_path)
    summary["model_id"] = model_id
    summary["vllm_served_name"] = _vllm_served_name(model_id, vllm_served_name)
    if hf_revision:
        summary["hf_revision"] = hf_revision
    summary["n_examples"] = n_examples
    summary["prompt_type"] = prompt_type
    summary["task_type"] = task_type
    summary["gpu_tier"] = gpu_tier
    if include_results_data:
        summary = attach_results_data(summary, results_path)
    return summary


def _checkpoint_context(
    model_id: str,
) -> tuple[str, str | None, str]:
    """Return ``(hf_model_id, revision, display_model_id)`` for eval/logging."""
    hf_model_id, revision = parse_checkpoint_model_id(model_id)
    display_model_id = model_id if revision else hf_model_id
    return hf_model_id, revision, display_model_id


def _run_model_eval(
    model_id: str,
    max_tasks: Optional[int],
    gpu_tier: str,
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: str | None = None,
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    original_model_id = model_id
    display_override: str | None = None
    if is_sft_checkpoint_model(model_id):
        if is_miniberta_sft_checkpoint(model_id):
            display_override = sft_run_id_from_model_id(model_id)
            print(
                f"Resolving MiniBERTa MLM SFT → Volume path "
                f"({display_override})...",
                flush=True,
            )
        model_id = _ensure_sft_checkpoint_dir(model_id)
    elif is_childes_model(model_id):
        display_override = normalize_childes_model_id(model_id)
        print(
            f"Resolving CHILDES ladder subfolder → local path "
            f"({display_override})...",
            flush=True,
        )
        model_id = resolve_childes_local_path(model_id)
    elif is_td_model(model_id):
        display_override = normalize_td_model_id(model_id)
        print(
            f"Resolving TinyDialogues checkpoint → local path "
            f"({display_override})...",
            flush=True,
        )
        model_id = resolve_td_local_path(model_id)
    elif is_os_model(model_id):
        display_override = normalize_os_model_id(model_id)
        print(
            f"Resolving OpenSubtitles checkpoint → local path "
            f"({display_override})...",
            flush=True,
        )
        model_id = resolve_os_local_path(model_id)
    elif is_wiki_model(model_id):
        display_override = normalize_wiki_model_id(model_id)
        print(
            f"Resolving Wiki checkpoint → local path "
            f"({display_override})...",
            flush=True,
        )
        model_id = resolve_wiki_local_path(model_id)
    elif is_ts_model(model_id):
        display_override = normalize_ts_model_id(model_id)
        print(
            f"Resolving TinyStories checkpoint → local path "
            f"({display_override})...",
            flush=True,
        )
        model_id = resolve_ts_local_path(model_id)

    hf_model_id, hf_revision, display_model_id = _checkpoint_context(model_id)
    if display_override is not None:
        display_model_id = display_override
    resolved_score_mode = _score_mode_for_model(model_id, score_mode)
    # Hub MiniBERTa and Volume MLM SFT both use HF PLL (not vLLM).
    if is_miniberta_model(hf_model_id) or is_miniberta_sft_checkpoint(
        original_model_id
    ):
        return _run_hf_model_eval(
            model_id,
            max_tasks,
            gpu_tier,
            n_examples=n_examples,
            prompt_type=prompt_type,
            ravens_prompt_mode=ravens_prompt_mode,
            task_type=task_type,
            score_mode=resolved_score_mode,
            ravens_tasks_json=ravens_tasks_json,
            display_model_id=display_model_id,
        )

    run_types = resolve_prompt_types(prompt_type)
    served = _vllm_served_name(model_id, vllm_served_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = CONTAINER_SRC
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    if len(run_types) == 1:
        pt = run_types[0]
        results_path = _run_script(
            hf_model_id,
            max_tasks,
            gpu_tier,
            vllm_served_name,
            n_examples,
            prompt_type=pt,
            ravens_prompt_mode=ravens_prompt_mode,
            hf_revision=hf_revision,
            task_type=task_type,
            score_mode=resolved_score_mode,
            ravens_tasks_json=ravens_tasks_json,
        )
        summary = _make_model_summary(
            results_path,
            model_id=display_model_id,
            prompt_type=pt,
            n_examples=n_examples,
            gpu_tier=gpu_tier,
            vllm_served_name=vllm_served_name,
            include_results_data=True,
            hf_revision=hf_revision,
            task_type=task_type,
        )
        print(json.dumps(public_summary(summary), indent=2), flush=True)
        hf_cache_volume.commit()
        return summary

    serve_cmd = _build_vllm_serve_cmd(hf_model_id, gpu_tier, revision=hf_revision)
    log_path = _vllm_log_path(display_model_id)
    proc, reader, log_file = _start_vllm_server(serve_cmd, log_path, env)
    try:
        _wait_for_vllm(proc, log_path, serve_cmd)
        summaries: dict[str, dict[str, Any]] = {}
        for pt in run_types:
            print(f"\n--- prompt_type={pt} ---\n", flush=True)
            results_path = _run_baby_reasoning_cli(
                served,
                max_tasks,
                n_examples,
                pt,
                env,
                ravens_prompt_mode if pt == "instruction" else "auto",
                task_type=task_type,
                score_mode=resolved_score_mode,
                ravens_tasks_json=ravens_tasks_json,
            )
            summary = _make_model_summary(
                results_path,
                model_id=display_model_id,
                prompt_type=pt,
                n_examples=n_examples,
                gpu_tier=gpu_tier,
                vllm_served_name=vllm_served_name,
                include_results_data=True,
                hf_revision=hf_revision,
                task_type=task_type,
            )
            summaries[pt] = summary
            print(json.dumps(public_summary(summary), indent=2), flush=True)
        hf_cache_volume.commit()
        return summaries
    finally:
        _stop_vllm_server(proc, log_file, reader)


def _ensure_sft_checkpoint_dir(model_id: str) -> str:
    """Resolve SFT Volume path and require the checkpoint directory to exist."""
    path = Path(resolve_sft_model_id(model_id))
    if not path.is_dir():
        raise FileNotFoundError(
            f"SFT checkpoint not found at {path}. "
            "Expected a directory on Volume 'ravens-babylm-sft' "
            f"(run_id={sft_run_id_from_model_id(model_id)!r}). "
            "Train with babylm_finetune/scripts/modal_sft.py first."
        )
    return str(path)


def _experiments_path_for_model(
    model_id: str,
    n_examples: int | None = None,
) -> Path:
    if is_sft_checkpoint_model(model_id):
        return babylm_sft_experiments_md(model_id, n_examples=n_examples)
    if is_pythia_checkpoint_model_id(model_id):
        return PYTHIA_CHECKPOINTS_MD_LOCAL
    if is_olmo2_checkpoint_model_id(model_id):
        return OLMO_CHECKPOINTS_MD_LOCAL
    if is_babylm_model(model_id):
        return BABYLM_EXPERIMENTS_MD_LOCAL
    if is_childes_model(model_id):
        return CHILDES_EXPERIMENTS_MD_LOCAL
    if is_td_model(model_id):
        return TD_EXPERIMENTS_MD_LOCAL
    if is_os_model(model_id):
        return OS_EXPERIMENTS_MD_LOCAL
    if is_wiki_model(model_id):
        return WIKI_EXPERIMENTS_MD_LOCAL
    if is_ts_model(model_id):
        return TS_EXPERIMENTS_MD_LOCAL
    return EXPERIMENTS_MD_LOCAL


def _sft_run_id_from_log_stem(stem: str) -> str:
    """Strip trailing ``__n{N}`` suffix from an SFT eval log filename stem."""
    import re

    return re.sub(r"__n\d+$", "", stem)


def _file_preamble_for_path(experiments_path: Path) -> str | None:
    try:
        if experiments_path.resolve().parent == BABYLM_FINETUNE_LOGS_DIR.resolve():
            run_id = _sft_run_id_from_log_stem(experiments_path.stem)
            return (
                f"# BabyLM SFT eval — `{run_id}`\n\n"
                f"Finetuned checkpoint on Volume `ravens-babylm-sft` "
                f"at `/checkpoints/{run_id}`.\n\n"
                "Auto-appended by `ravens-modal` / `modal_eval.py` for SFT run_ids.\n\n"
            )
    except OSError:
        pass
    if experiments_path == BABYLM_EXPERIMENTS_MD_LOCAL:
        return BABYLM_EXPERIMENTS_PREAMBLE
    if experiments_path == CHILDES_EXPERIMENTS_MD_LOCAL:
        return CHILDES_EXPERIMENTS_PREAMBLE
    if experiments_path == TD_EXPERIMENTS_MD_LOCAL:
        return TD_EXPERIMENTS_PREAMBLE
    if experiments_path == OS_EXPERIMENTS_MD_LOCAL:
        return OS_EXPERIMENTS_PREAMBLE
    if experiments_path == WIKI_EXPERIMENTS_MD_LOCAL:
        return WIKI_EXPERIMENTS_PREAMBLE
    if experiments_path == TS_EXPERIMENTS_MD_LOCAL:
        return TS_EXPERIMENTS_PREAMBLE
    if experiments_path == PYTHIA_CHECKPOINTS_MD_LOCAL:
        return PYTHIA_CHECKPOINTS_PREAMBLE
    if experiments_path == OLMO_CHECKPOINTS_MD_LOCAL:
        return OLMO_CHECKPOINTS_PREAMBLE
    return None


def _eval_data_settings_fragment(
    n_examples: int,
    task_type: str,
    ravens_tasks_json: str | None = None,
) -> str | None:
    """Human-readable eval-data note for experiment logs (Raven ICL test sets)."""
    if ravens_tasks_json:
        name = Path(ravens_tasks_json).name
        return f"eval data `{name}` (`--n-examples {n_examples}`)"
    if task_type == "ravens" and n_examples in _RAVENS_ICL_TEST_BY_N:
        name = _RAVENS_ICL_TEST_BY_N[n_examples]
        return f"eval data `{name}` (`--n-examples {n_examples}`)"
    return None


def _log_experiment_local(
    *,
    model_id: str,
    run_label: str,
    model_summaries: dict[str, dict[str, Any]],
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    accuracy_delta: Optional[float] = None,
    ravens_tasks_json: str | None = None,
) -> None:
    """Append results to the local experiments log (after ``.remote()``)."""
    from ravens_numerical.analysis.experiment_log import append_experiment_entry

    resolved = _score_mode_for_model(model_id, score_mode)
    experiments_path = _experiments_path_for_model(model_id, n_examples=n_examples)
    settings_parts = [
        f"{_experiment_settings_note(model_id, resolved)}; `--n-examples {n_examples}`",
        f"`--task-type {task_type}`",
    ]
    eval_data_note = _eval_data_settings_fragment(
        n_examples, task_type, ravens_tasks_json
    )
    if eval_data_note:
        settings_parts.append(eval_data_note)
    if task_type in RAVENS_LIKE_TASK_TYPES:
        settings_parts.append(f"`--prompt-type {prompt_type}`")
        if prompt_type == "instruction":
            settings_parts.append(
                f"`--ravens-prompt-mode {resolve_instruction_prompt_mode(ravens_prompt_mode)}`"
            )
    settings_note = "; ".join(settings_parts)
    append_experiment_entry(
        experiments_path,
        run_label=run_label,
        model_summaries=model_summaries,
        max_tasks=max_tasks,
        accuracy_delta=accuracy_delta,
        settings_note=settings_note,
        file_preamble=_file_preamble_for_path(experiments_path),
        step=checkpoint_step_from_model_id(model_id),
        n_examples=n_examples,
    )


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=3600,
    volumes=EVAL_VOLUMES,
)
def run_ravens_eval_t4(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate one HF model via in-container vLLM (T4)."""
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="T4",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        task_type=task_type,
        score_mode=score_mode,
        ravens_tasks_json=ravens_tasks_json,
    )


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=7200,
    volumes=EVAL_VOLUMES,
)
def run_ravens_eval_a10g(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate one HF model via in-container vLLM (A10G)."""
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="A10G",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        task_type=task_type,
        score_mode=score_mode,
        ravens_tasks_json=ravens_tasks_json,
    )


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=7200,
    volumes=EVAL_VOLUMES,
)
def run_ravens_eval_a100(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate one HF model via in-container vLLM (A100).

    Used for ≥10B models (e.g. Pythia-12B, Qwen3-14B) that OOM on A10G in fp16.
    """
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="A100",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
        task_type=task_type,
        score_mode=score_mode,
        ravens_tasks_json=ravens_tasks_json,
    )


def _remote_run_ravens_eval(
    model_id: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: Optional[str] = None,
) -> dict[str, Any]:
    tier = gpu_tier_for_model(model_id)
    kwargs = {
        "model_id": model_id,
        "max_tasks": max_tasks,
        "n_examples": n_examples,
        "prompt_type": prompt_type,
        "ravens_prompt_mode": ravens_prompt_mode,
        "task_type": task_type,
        "score_mode": score_mode,
        "ravens_tasks_json": ravens_tasks_json,
    }
    if tier == "T4":
        return run_ravens_eval_t4.remote(**kwargs)
    if tier == "A100":
        return run_ravens_eval_a100.remote(**kwargs)
    return run_ravens_eval_a10g.remote(**kwargs)


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=3600,
    volumes=EVAL_VOLUMES,
)
def run_pythia_ravens(
    max_tasks: Optional[int] = None,
    model_id: str = DEFAULT_PYTHIA_VLLM,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    """Evaluate Pythia on ravens_numerical via in-container vLLM."""
    return _run_model_eval(model_id, max_tasks, gpu_tier="T4", n_examples=n_examples)


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=3600,
    volumes=EVAL_VOLUMES,
)
def run_babylm_ravens(
    max_tasks: Optional[int] = None,
    model_id: str = DEFAULT_BABYLM_VLLM,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    """Evaluate BabyLM GPT-2 baseline on ravens_numerical via in-container vLLM."""
    return _run_model_eval(model_id, max_tasks, gpu_tier="T4", n_examples=n_examples)


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=7200,
    volumes=EVAL_VOLUMES,
)
def run_miniberta_ravens(
    max_tasks: Optional[int] = None,
    model_id: str = DEFAULT_MINIBERTA_VLLM,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any]:
    """Evaluate MiniBERTa RoBERTa on ravens_numerical via in-container HuggingFace."""
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="T4",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
    )


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=7200,
    volumes=EVAL_VOLUMES,
)
def run_qwen3_ravens(
    max_tasks: Optional[int] = None,
    model_id: str = DEFAULT_QWEN3_VLLM,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    """Evaluate Qwen3 (HF id) on ravens_numerical via in-container vLLM."""
    return _run_model_eval(model_id, max_tasks, gpu_tier="A10G", n_examples=n_examples)


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=10800,
    volumes=EVAL_VOLUMES,
)
def run_compare_ravens(
    max_tasks: Optional[int] = None,
    pythia_model_id: str = DEFAULT_PYTHIA_VLLM,
    qwen3_model_id: str = DEFAULT_QWEN3_VLLM,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    """Run Pythia then Qwen3 sequentially; return both summaries for comparison."""
    pythia = _run_model_eval(
        pythia_model_id, max_tasks, gpu_tier="A10G", n_examples=n_examples
    )
    qwen3 = _run_model_eval(
        qwen3_model_id, max_tasks, gpu_tier="A10G", n_examples=n_examples
    )
    comparison = {
        "pythia": pythia,
        "qwen3": qwen3,
        "n_examples": n_examples,
        "accuracy_delta_qwen3_minus_pythia": qwen3["accuracy"] - pythia["accuracy"],
    }
    print(json.dumps(comparison, indent=2), flush=True)
    return comparison


@app.local_entrypoint()
def main(
    max_tasks: Optional[int] = None,
    models: str = "sweep",
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
    ravens_tasks_json: Optional[str] = None,
    combined_sft_log: Optional[str] = None,
    combined_sft_results_json: Optional[str] = None,
) -> None:
    """Local entry: ``modal run -m ravens_numerical.cloud.modal_eval --models sweep``.

    ``--models`` accepts a HuggingFace id, comma-separated ids, or aliases:
    ``sweep`` (full scaling ladder), ``pythia``, ``pythia-checkpoints``,
    ``olmo`` / ``olmo2``, ``olmo-checkpoints`` / ``olmo2-checkpoints``,
    ``qwen3``, ``babylm``, ``miniberta``, ``babylm-sft`` (Volume SFT
    ``run_id``s starting with ``babylm-`` → combined
    ``babylm_finetune/logs/all_sft_evals.md``), ``babylm-sft-alltasks``
    (``babylm-`` run_ids with ``__all_types__`` only →
    ``babylm_finetune/logs/babylm_sft_alltasks_evals.md``), ``childes-sft`` (``run_id``s
    starting with ``childes-`` → ``babylm_finetune/logs/childes_sft_evals.md``),
    ``td-sft`` (``run_id``s starting with ``td-`` →
    ``babylm_finetune/logs/td_sft_evals.md``), ``os-sft`` (``run_id``s starting
    with ``os-`` → ``babylm_finetune/logs/os_sft_evals.md``), ``wiki-sft``
    (``run_id``s starting with ``wiki-`` →
    ``babylm_finetune/logs/wiki_sft_evals.md``), ``ts-sft`` (``run_id``s
    starting with ``ts-`` → ``babylm_finetune/logs/ts_sft_evals.md``),
    ``miniberta-sft`` (``run_id``s starting with ``miniberta-`` →
    ``babylm_finetune/logs/miniberta_sft_evals.md``; HF PLL).

    ``--combined-sft-log`` overrides the combined SFT markdown path (default
    ``all_sft_evals.md`` / ``babylm_sft_alltasks_evals.md`` /
    ``childes_sft_evals.md`` / ``td_sft_evals.md`` /
    ``os_sft_evals.md`` / ``wiki_sft_evals.md`` / ``ts_sft_evals.md`` /
    ``miniberta_sft_evals.md`` when a multi-SFT batch is written).

    ``--combined-sft-results-json`` optionally writes a plot-friendly JSON snapshot
    of the same run_id → accuracy table.

    For ``--task-type ravens``, ``--n-examples`` also selects the test JSON:
    ``0`` → zero-shot ``complete.json``; ``1`` → ``oneICL_test.json``; ``3`` →
    ``threeICL_test.json`` (disjoint ICL bank baked into the file).

    ``--ravens-tasks-json`` overrides that file selection (repo path or alias
    ``complete`` → ``data/complete.json`` (default, 500 items);
    ``tasks`` / ``tasks.json`` → ``data/tasks.json`` (7-type 350-item backup);
    ``5digit`` / ``tasks_5digit`` → ``data/tasks_5digit.json``;
    ``challenge`` → ``data/challenge_tasks.json``, 150 harder items).

    ``--task-type`` is ``ravens`` (default; ``complete.json`` / ravens_numerical),
    ``webb`` (original Webb digit matrices / ``tasks_webb.json``, same harness),
    ``aba`` (ABA/ABB rules from ``tasks_aba.json``), or ``hierarchical``
    (hierarchical equality from ``tasks_aba.json``).

    ``--score-mode`` is ``forced_choice`` (default; length-normalized echo
    logprob-argmax ``correct`` for ravens / webb / aba / matrix, with generation
    match only when echo abstains; hierarchical also uses vLLM structured choice
    during generation), ``free_gen``, or ``auto`` (``forced_choice`` for Pythia
    ``@stepN`` checkpoints, else free_gen).

    ``--prompt-type`` is ``completion`` (bracket fill-in; default for ravens / webb),
    ``instruction`` (letter MCQ), or ``both`` (runs each sequentially in one vLLM
    session per model; MiniBERTa uses HF without vLLM). Ignored for
    ``--task-type aba`` / ``hierarchical``. With ``forced_choice``, ravens / webb
    always use completion.

    Appends results to ``artifacts/logs/experiments.md`` after each model
    (BabyLM models → ``artifacts/logs/babyLMexperiments.md``; CHILDES ladder →
    ``artifacts/logs/childesExperiments.md``; TinyDialogues →
    ``artifacts/logs/tdExperiments.md``; OpenSubtitles →
    ``artifacts/logs/opensubtitlesExperiments.md``; Wiki →
    ``artifacts/logs/wikiExperiments.md``; TinyStories →
    ``artifacts/logs/tinystoriesExperiments.md``; Pythia checkpoints →
    ``artifacts/logs/pythiacheckpoints.md``; OLMo 2 checkpoints →
    ``artifacts/logs/olmocheckpoints.md``; MiniBERTa → ``experiments.md``;
    SFT Volume checkpoints → ``babylm_finetune/logs/<run_id>.md``).

    Per-trial JSON is saved locally under ``artifacts/runs/`` (same layout
    as in-container paths) for downstream audit.

    ``--max-tasks`` defaults to all tasks in ``data/complete.json`` (currently 500).
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"--task-type must be 'ravens', 'webb', 'aba', or 'hierarchical', got {task_type!r}"
        )
    if prompt_type not in PROMPT_TYPE_CLI:
        raise ValueError(
            f"--prompt-type must be one of {PROMPT_TYPE_CLI}, got {prompt_type!r}"
        )
    if score_mode not in SCORE_MODES:
        raise ValueError(
            f"--score-mode must be one of {SCORE_MODES}, got {score_mode!r}"
        )

    container_tasks_json = resolve_container_ravens_tasks_json(
        task_type=task_type,
        n_examples=n_examples,
        ravens_tasks_json=ravens_tasks_json,
    )
    if container_tasks_json is not None:
        print(
            f"Using Raven tasks JSON: {container_tasks_json}",
            flush=True,
        )

    models_key = models.strip().lower()
    batch_all_sft = models_key in ("babylm-sft", "babylm_sft", "sft")
    batch_babylm_sft_alltasks = models_key in (
        "babylm-sft-alltasks",
        "babylm_sft_alltasks",
        "babylm-sft-all-types",
        "babylm_sft_all_types",
    )
    batch_childes_sft = models_key in ("childes-sft", "childes_sft")
    batch_td_sft = models_key in ("td-sft", "td_sft")
    batch_os_sft = models_key in ("os-sft", "os_sft", "opensubtitles-sft")
    batch_wiki_sft = models_key in ("wiki-sft", "wiki_sft")
    batch_ts_sft = models_key in ("ts-sft", "ts_sft", "tinystories-sft")
    batch_miniberta_sft = models_key in ("miniberta-sft", "miniberta_sft")
    if batch_all_sft:
        model_ids = resolve_babylm_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} BabyLM SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_babylm_sft_alltasks:
        model_ids = resolve_babylm_sft_alltasks_models_arg()
        print(
            f"Evaluating {len(model_ids)} BabyLM all_types SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_childes_sft:
        model_ids = resolve_childes_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} CHILDES SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_td_sft:
        model_ids = resolve_td_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} TinyDialogues SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_os_sft:
        model_ids = resolve_os_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} OpenSubtitles SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_wiki_sft:
        model_ids = resolve_wiki_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} Wiki SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_ts_sft:
        model_ids = resolve_ts_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} TinyStories SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    elif batch_miniberta_sft:
        model_ids = resolve_miniberta_sft_models_arg()
        print(
            f"Evaluating {len(model_ids)} MiniBERTa MLM SFT checkpoint(s): "
            + ", ".join(sft_run_id_from_model_id(m) for m in model_ids),
            flush=True,
        )
    else:
        model_ids = resolve_models_arg(models)
    if not model_ids:
        raise ValueError(f"No models resolved from --models {models!r}")

    all_results: dict[str, Any] = {}
    sft_combined: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        resolved_score = _score_mode_for_model(model_id, score_mode)
        # ABA / hierarchical use completion-style prompts; Raven prompt-type is N/A.
        # Ravens / Webb forced_choice uses completion echo logprob argmax (not instruction JSON).
        if task_type in BABY_TASK_TYPES:
            effective_prompt_type = "completion"
        elif resolved_score == "forced_choice" and task_type in RAVENS_LIKE_TASK_TYPES:
            effective_prompt_type = "completion"
        else:
            effective_prompt_type = prompt_type
        print(
            f"\n=== Evaluating {model_id} (GPU {gpu_tier_for_model(model_id)}, "
            f"task_type={task_type}, prompt={effective_prompt_type}, "
            f"score_mode={resolved_score}) ===\n",
            flush=True,
        )
        summary = _remote_run_ravens_eval(
            model_id,
            max_tasks,
            n_examples,
            prompt_type=effective_prompt_type,
            ravens_prompt_mode=ravens_prompt_mode,
            task_type=task_type,
            score_mode=score_mode,
            ravens_tasks_json=container_tasks_json,
        )
        saved_paths = export_run_results(
            summary,
            effective_prompt_type,
            local_runs_root=LOCAL_RUNS_DIR,
            container_runs_root=CONTAINER_RUNS_DIR,
        )
        for path in saved_paths:
            print(f"Saved results → {path}", flush=True)

        safe_key = model_id.replace("/", "--")

        if effective_prompt_type == "both":
            assert isinstance(summary, dict)
            all_results[model_id] = {
                pt: public_summary(pt_summary)
                for pt, pt_summary in summary.items()
                if isinstance(pt_summary, dict)
            }
            for pt, pt_summary in summary.items():
                _log_experiment_local(
                    model_id=model_id,
                    run_label=f"run_ravens_eval ({safe_key}, {pt})",
                    model_summaries={safe_key: public_summary(pt_summary)},
                    max_tasks=max_tasks,
                    n_examples=n_examples,
                    prompt_type=pt,
                    ravens_prompt_mode=ravens_prompt_mode,
                    task_type=task_type,
                    score_mode=score_mode,
                    ravens_tasks_json=container_tasks_json,
                )
        else:
            assert isinstance(summary, dict) and "prompt_type" in summary
            pub = public_summary(summary)
            all_results[model_id] = pub
            _log_experiment_local(
                model_id=model_id,
                run_label=f"run_ravens_eval ({safe_key}, {task_type})",
                model_summaries={safe_key: pub},
                max_tasks=max_tasks,
                n_examples=n_examples,
                prompt_type=effective_prompt_type,
                ravens_prompt_mode=ravens_prompt_mode,
                task_type=task_type,
                score_mode=score_mode,
                ravens_tasks_json=container_tasks_json,
            )
            if is_sft_checkpoint_model(model_id):
                sft_combined[sft_run_id_from_model_id(model_id)] = pub

    write_combined = (
        batch_all_sft
        or batch_babylm_sft_alltasks
        or batch_childes_sft
        or batch_td_sft
        or batch_os_sft
        or batch_wiki_sft
        or batch_ts_sft
        or batch_miniberta_sft
        or (len(sft_combined) > 1 and len(sft_combined) == len(model_ids))
    )
    if write_combined:
        from ravens_numerical.analysis.experiment_log import write_all_sft_evals_md

        eval_data_note = _eval_data_settings_fragment(
            n_examples, task_type, container_tasks_json
        )
        if batch_miniberta_sft:
            settings = (
                "Modal HF transformers (`modal_eval.py`); PLL scoring; "
                f"`--score-mode forced_choice`; `--n-examples {n_examples}`; "
                f"`--task-type {task_type}`; `--prompt-type completion`"
                + (f"; {eval_data_note}" if eval_data_note else "")
            )
        else:
            settings = (
                "Modal vLLM (`modal_eval.py`); `--score-mode forced_choice`; "
                f"`--n-examples {n_examples}`; `--task-type {task_type}`; "
                "`--prompt-type completion`"
                + (f"; {eval_data_note}" if eval_data_note else "")
            )
        if batch_miniberta_sft:
            combined_default = MINIBERTA_SFT_ALL_EVALS_MD
        elif batch_ts_sft:
            combined_default = TS_SFT_ALL_EVALS_MD
        elif batch_wiki_sft:
            combined_default = WIKI_SFT_ALL_EVALS_MD
        elif batch_os_sft:
            combined_default = OS_SFT_ALL_EVALS_MD
        elif batch_td_sft:
            combined_default = TD_SFT_ALL_EVALS_MD
        elif batch_childes_sft:
            combined_default = CHILDES_SFT_ALL_EVALS_MD
        elif batch_babylm_sft_alltasks:
            combined_default = BABYLM_SFT_ALLTASKS_EVALS_MD
        else:
            combined_default = BABYLM_SFT_ALL_EVALS_MD
        combined_md = combined_sft_log_path_for_n_examples(
            Path(combined_sft_log) if combined_sft_log else None,
            n_examples,
            default=combined_default,
        )
        if not combined_md.is_absolute():
            combined_md = REPO_ROOT / combined_md
        out_path = write_all_sft_evals_md(
            combined_md,
            results_by_run_id=sft_combined,
            settings_note=settings,
            max_tasks=max_tasks,
            n_examples=n_examples,
        )
        print(f"Wrote combined SFT eval log → {out_path}", flush=True)

        if combined_sft_results_json:
            results_path = Path(combined_sft_results_json)
            if not results_path.is_absolute():
                results_path = REPO_ROOT / results_path
            json_path = _write_combined_sft_results_json(
                results_path,
                results_by_run_id=sft_combined,
                n_examples=n_examples,
            )
            print(f"Wrote combined SFT results JSON → {json_path}", flush=True)

    output = {
        "models": model_ids,
        "n_examples": n_examples,
        "task_type": task_type,
        "prompt_type": effective_prompt_type,
        "ravens_prompt_mode": ravens_prompt_mode,
        "score_mode": score_mode,
        "max_tasks": max_tasks,
        "ravens_tasks_json": container_tasks_json,
        "results": all_results,
    }
    print(json.dumps(output, indent=2))
