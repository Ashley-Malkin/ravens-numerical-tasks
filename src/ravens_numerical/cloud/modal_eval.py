#!/usr/bin/env python3
"""Run ravens_numerical eval on Modal (Option A): vLLM server inside the GPU container.

Compare Pythia vs Qwen3 (or a size ladder) on the same ``tasks.json`` / prompts as
``baby_reasoning_eval`` locally, using ``script/run --backend vllm`` against ``http://127.0.0.1:8000``.

Setup (once on your machine):

    pip install modal
    modal setup

Smoke test:

    modal run baby_reasoning_eval/modal_eval.py --max-tasks 10 --models EleutherAI/pythia-70m-deduped

Full scaling ladder (140 tasks; appends to ``experiments.md``; BabyLM → ``babyLMexperiments.md``;
Pythia checkpoints → ``pythiacheckpoints.md``; OLMo 2 checkpoints → ``olmocheckpoints.md``;
saves per-trial JSON under ``artifacts/runs/``):

    modal run -m ravens_numerical.cloud.modal_eval --models sweep --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models pythia-checkpoints --n-examples 0
    modal run -m ravens_numerical.cloud.modal_eval --models olmo2-checkpoints --n-examples 0

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
    CONTAINER_RAVENS_ROOT,
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PYTHIA_CHECKPOINTS_MD,
    REPO_ROOT,
    RUNS_DIR,
)

# Paths inside the Modal container (repo copied here via ``add_local_dir``).
CONTAINER_TASKS_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks.json"
CONTAINER_TASKS_ABA_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks_aba.json"
CONTAINER_TASKS_WEBB_JSON = f"{CONTAINER_RAVENS_ROOT}/data/tasks_webb.json"
CONTAINER_RUNS_DIR = f"{CONTAINER_RAVENS_ROOT}/artifacts/runs"
CONTAINER_SRC = f"{CONTAINER_RAVENS_ROOT}/src"

EASY_TASK_DIRS = {
    "aba": ("rules",),
    "hierarchical": ("hierarchical",),
}
BABY_TASK_TYPES = frozenset(EASY_TASK_DIRS)
RAVENS_LIKE_TASK_TYPES = frozenset({"ravens", "webb"})
SUPPORTED_TASK_TYPES = frozenset({"ravens", "webb", *BABY_TASK_TYPES})

EXPERIMENTS_MD_LOCAL = EXPERIMENTS_MD
BABYLM_EXPERIMENTS_MD_LOCAL = BABYLM_EXPERIMENTS_MD
PYTHIA_CHECKPOINTS_MD_LOCAL = PYTHIA_CHECKPOINTS_MD
OLMO_CHECKPOINTS_MD_LOCAL = OLMO_CHECKPOINTS_MD
LOCAL_RUNS_DIR = RUNS_DIR

BABYLM_EXPERIMENTS_PREAMBLE = (
    "# BabyLM Ravens numerical experiments (Modal / vLLM)\n\n"
    "Auto-appended by `ravens-modal` for `--models babylm` "
    "and other BabyLM HuggingFace ids.\n\n"
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
    is_miniberta_model,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    is_qwen3_instruct_model,
    is_qwen3_model,
    max_model_len_for_model,
    parse_checkpoint_model_id,
    resolve_instruction_prompt_mode,
    resolve_models_arg,
)

def _experiment_settings_note(model_id: str, score_mode: str | None = None) -> str:
    hf_model_id, revision = parse_checkpoint_model_id(model_id)
    if is_miniberta_model(hf_model_id):
        max_len = max_model_len_for_model(hf_model_id)
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
]

app = modal.App("ravens-baby-reasoning-vllm")

hf_cache_volume = modal.Volume.from_name("ravens-hf-cache", create_if_missing=True)

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
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """MiniBERTa eval via HuggingFace transformers (no vLLM server)."""
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
        )
        summary = _make_model_summary(
            results_path,
            model_id=model_id,
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
        )
        summary = _make_model_summary(
            results_path,
            model_id=model_id,
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
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    hf_model_id, hf_revision, display_model_id = _checkpoint_context(model_id)
    resolved_score_mode = _score_mode_for_model(model_id, score_mode)
    if is_miniberta_model(hf_model_id):
        return _run_hf_model_eval(
            hf_model_id,
            max_tasks,
            gpu_tier,
            n_examples=n_examples,
            prompt_type=prompt_type,
            ravens_prompt_mode=ravens_prompt_mode,
            task_type=task_type,
            score_mode=resolved_score_mode,
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


def _experiments_path_for_model(model_id: str) -> Path:
    if is_pythia_checkpoint_model_id(model_id):
        return PYTHIA_CHECKPOINTS_MD_LOCAL
    if is_olmo2_checkpoint_model_id(model_id):
        return OLMO_CHECKPOINTS_MD_LOCAL
    if is_babylm_model(model_id):
        return BABYLM_EXPERIMENTS_MD_LOCAL
    return EXPERIMENTS_MD_LOCAL


def _file_preamble_for_path(experiments_path: Path) -> str | None:
    if experiments_path == BABYLM_EXPERIMENTS_MD_LOCAL:
        return BABYLM_EXPERIMENTS_PREAMBLE
    if experiments_path == PYTHIA_CHECKPOINTS_MD_LOCAL:
        return PYTHIA_CHECKPOINTS_PREAMBLE
    if experiments_path == OLMO_CHECKPOINTS_MD_LOCAL:
        return OLMO_CHECKPOINTS_PREAMBLE
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
) -> None:
    """Append results to the local experiments log (after ``.remote()``)."""
    from ravens_numerical.analysis.experiment_log import append_experiment_entry

    resolved = _score_mode_for_model(model_id, score_mode)
    experiments_path = _experiments_path_for_model(model_id)
    settings_parts = [
        f"{_experiment_settings_note(model_id, resolved)}; `--n-examples {n_examples}`",
        f"`--task-type {task_type}`",
    ]
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
    )


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def run_ravens_eval_t4(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
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
    )


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def run_ravens_eval_a10g(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
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
    )


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def run_ravens_eval_a100(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
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
    )


def _remote_run_ravens_eval(
    model_id: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
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
    volumes={"/root/.cache/huggingface": hf_cache_volume},
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
    volumes={"/root/.cache/huggingface": hf_cache_volume},
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
    volumes={"/root/.cache/huggingface": hf_cache_volume},
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
    volumes={"/root/.cache/huggingface": hf_cache_volume},
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
    volumes={"/root/.cache/huggingface": hf_cache_volume},
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
    max_tasks: Optional[int] = 140,
    models: str = "sweep",
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "completion",
    ravens_prompt_mode: str = "auto",
    task_type: str = "ravens",
    score_mode: str = "forced_choice",
) -> None:
    """Local entry: ``modal run -m ravens_numerical.cloud.modal_eval --models sweep``.

    ``--models`` accepts a HuggingFace id, comma-separated ids, or aliases:
    ``sweep`` (full scaling ladder), ``pythia``, ``pythia-checkpoints``,
    ``olmo`` / ``olmo2``, ``olmo-checkpoints`` / ``olmo2-checkpoints``,
    ``qwen3``, ``babylm``, ``miniberta``.

    ``--task-type`` is ``ravens`` (default; ``tasks.json`` / ravens_numerical),
    ``webb`` (original Webb digit matrices / ``tasks_webb.json``, same harness),
    ``aba`` (ABA/ABB rules from ``tasks_aba.json``), or ``hierarchical``
    (hierarchical equality from ``tasks_aba.json``).

    ``--score-mode`` is ``forced_choice`` (default; echo logprob-argmax ``correct``
    for ravens / webb / aba / matrix; hierarchical also uses vLLM structured choice
    during generation), ``free_gen``, or ``auto`` (``forced_choice`` for Pythia
    ``@stepN`` checkpoints, else free_gen).

    ``--prompt-type`` is ``completion`` (bracket fill-in; default for ravens / webb),
    ``instruction`` (letter MCQ), or ``both`` (runs each sequentially in one vLLM
    session per model; MiniBERTa uses HF without vLLM). Ignored for
    ``--task-type aba`` / ``hierarchical``. With ``forced_choice``, ravens / webb
    always use completion.

    Appends results to ``artifacts/logs/experiments.md`` after each model
    (BabyLM models → ``artifacts/logs/babyLMexperiments.md``; Pythia checkpoints →
    ``artifacts/logs/pythiacheckpoints.md``; OLMo 2 checkpoints →
    ``artifacts/logs/olmocheckpoints.md``; MiniBERTa → ``experiments.md``).

    Per-trial JSON is saved locally under ``artifacts/runs/`` (same layout
    as in-container paths) for downstream audit.
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

    model_ids = resolve_models_arg(models)
    if not model_ids:
        raise ValueError(f"No models resolved from --models {models!r}")

    all_results: dict[str, Any] = {}
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
                )
        else:
            assert isinstance(summary, dict) and "prompt_type" in summary
            all_results[model_id] = public_summary(summary)
            _log_experiment_local(
                model_id=model_id,
                run_label=f"run_ravens_eval ({safe_key}, {task_type})",
                model_summaries={safe_key: public_summary(summary)},
                max_tasks=max_tasks,
                n_examples=n_examples,
                prompt_type=effective_prompt_type,
                ravens_prompt_mode=ravens_prompt_mode,
                task_type=task_type,
                score_mode=score_mode,
            )

    output = {
        "models": model_ids,
        "n_examples": n_examples,
        "task_type": task_type,
        "prompt_type": effective_prompt_type,
        "ravens_prompt_mode": ravens_prompt_mode,
        "score_mode": score_mode,
        "max_tasks": max_tasks,
        "results": all_results,
    }
    print(json.dumps(output, indent=2))
