#!/usr/bin/env python3
"""Run ravens_numerical eval on Modal (Option A): vLLM server inside the GPU container.

Compare Pythia vs Qwen3 (or a size ladder) on the same ``tasks.json`` / prompts as
``baby_reasoning_eval`` locally, using ``script/run --backend vllm`` against ``http://127.0.0.1:8000``.

Setup (once on your machine):

    pip install modal
    modal setup

Smoke test:

    modal run baby_reasoning_eval/modal_eval.py --max-tasks 10 --models EleutherAI/pythia-70m-deduped

Full scaling ladder (120 tasks; appends to ``experiments.md``):

    modal run baby_reasoning_eval/modal_eval.py --models sweep --n-examples 0

Direct remote functions (``::run_ravens_eval_t4``, etc.) skip ``experiments.md`` logging; use the entrypoint above.
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
REPO_ROOT = _MODAL_EVAL_DIR.parent
BABY_REASONING_ROOT = _MODAL_EVAL_DIR / "baby-reasoning"
EXPERIMENTS_MD_LOCAL = _MODAL_EVAL_DIR / "experiments.md"

# Paths inside the Modal container (repo copied here via ``add_local_dir``).
CONTAINER_RAVENS_ROOT = "/root/ravens"
CONTAINER_BABY_ROOT = f"{CONTAINER_RAVENS_ROOT}/baby_reasoning_eval/baby-reasoning"
CONTAINER_TASKS_JSON = f"{CONTAINER_RAVENS_ROOT}/tasks.json"


def _ensure_ravens_repo_on_path() -> None:
    """Make ``ravens_eval_models`` importable locally and in Modal containers."""
    candidates = [REPO_ROOT, Path(CONTAINER_RAVENS_ROOT)]
    for root in candidates:
        if (root / "ravens_eval_models.py").is_file():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return
    raise ImportError(
        "Could not find ravens_eval_models.py; expected under repo root "
        f"({REPO_ROOT}) or {CONTAINER_RAVENS_ROOT}"
    )


_ensure_ravens_repo_on_path()

from ravens_eval_models import (  # noqa: E402
    DEFAULT_PYTHIA_VLLM,
    DEFAULT_QWEN3_VLLM,
    gpu_tier_for_model,
    is_qwen3_model,
    max_model_len_for_model,
    resolve_instruction_prompt_mode,
    resolve_models_arg,
)

EXPERIMENT_SETTINGS_NOTE = (
    "Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; "
    "T4→TRITON_ATTN, A10G→FLASH_ATTN; `temperature=0`"
)

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
) -> str:
    if prompt_type == "completion":
        return f"{n_examples}_examples_completion.json"
    if prompt_mode == "choice_only":
        return f"{n_examples}_examples_choice_only.json"
    return f"{n_examples}_examples.json"

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

IGNORE_COPY = [
    "**/.venv/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/results/**",
    "**/*.egg-info/**",
    "**/node_modules/**",
    "**/.pytest_cache/**",
    "**/experiments.md",
]

app = modal.App("ravens-baby-reasoning-vllm")

hf_cache_volume = modal.Volume.from_name("ravens-hf-cache", create_if_missing=True)

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.18.0",
        "requests>=2.31",
        "tyro>=0.9",
        "pandas>=2.3",
    )
    .env({"PYTHONPATH": CONTAINER_RAVENS_ROOT})
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


def _build_vllm_serve_cmd(model_id: str, gpu_tier: str) -> list[str]:
    max_len = max_model_len_for_model(model_id)
    cmd = [
        "vllm",
        "serve",
        model_id,
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
    if is_qwen3_model(model_id):
        cmd.extend(
            [
                "--reasoning-parser",
                "qwen3",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        )
    return cmd


def _run_baby_reasoning_cli(
    served: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str,
    env: dict[str, str],
    ravens_prompt_mode: str = "auto",
) -> Path:
    """Run baby-reasoning CLI for one prompt type (vLLM must already be up)."""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"prompt_type must be one of {PROMPT_TYPES}, got {prompt_type!r}")

    cmd = [
        "python",
        f"{CONTAINER_BABY_ROOT}/script/run",
        "--backend",
        "vllm",
        "--base-url",
        VLLM_BASE_URL,
        "--models",
        served,
        "--tasks",
        "ravens_numerical",
        "--ravens-repo-root",
        CONTAINER_RAVENS_ROOT,
        "--ravens-tasks-json",
        CONTAINER_TASKS_JSON,
        "--n-examples",
        str(n_examples),
        "--ravens-prompt-type",
        prompt_type,
    ]
    if prompt_type == "instruction":
        cmd.extend(["--ravens-prompt-mode", ravens_prompt_mode])
    if max_tasks is not None:
        cmd.extend(["--ravens-max-tasks", str(max_tasks)])

    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=CONTAINER_BABY_ROOT, env=env)

    results_root = Path(CONTAINER_BABY_ROOT) / "results"
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


def _run_script(
    model_id: str,
    max_tasks: Optional[int],
    gpu_tier: str,
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> Path:
    """Start vLLM, run baby-reasoning CLI, return path to results JSON."""
    run_types = resolve_prompt_types(prompt_type)
    if len(run_types) != 1:
        raise ValueError(
            f"_run_script expects a single prompt type; got {prompt_type!r}. "
            "Use _run_model_eval for ``both``."
        )
    pt = run_types[0]
    served = vllm_served_name or model_id
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CONTAINER_RAVENS_ROOT}:{CONTAINER_BABY_ROOT}"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    serve_cmd = _build_vllm_serve_cmd(model_id, gpu_tier)
    log_path = _vllm_log_path(model_id)
    proc, reader, log_file = _start_vllm_server(serve_cmd, log_path, env)
    try:
        _wait_for_vllm(proc, log_path, serve_cmd)
        return _run_baby_reasoning_cli(
            served, max_tasks, n_examples, pt, env, ravens_prompt_mode
        )
    finally:
        _stop_vllm_server(proc, log_file, reader)


def _summarize_results(results_path: Path) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected results shape in {results_path}")

    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in data:
        correct = bool(row.get("score", {}).get("correct", False))
        task_type = (
            row.get("stimulus", {})
            .get("metadata", {})
            .get("task", {})
            .get("task_type", "unknown")
        )
        by_type[str(task_type)].append(correct)

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
) -> dict[str, Any]:
    summary = _summarize_results(results_path)
    summary["model_id"] = model_id
    summary["vllm_served_name"] = vllm_served_name or model_id
    summary["n_examples"] = n_examples
    summary["prompt_type"] = prompt_type
    summary["gpu_tier"] = gpu_tier
    return summary


def _run_model_eval(
    model_id: str,
    max_tasks: Optional[int],
    gpu_tier: str,
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    run_types = resolve_prompt_types(prompt_type)
    served = vllm_served_name or model_id
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CONTAINER_RAVENS_ROOT}:{CONTAINER_BABY_ROOT}"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    if len(run_types) == 1:
        pt = run_types[0]
        results_path = _run_script(
            model_id,
            max_tasks,
            gpu_tier,
            vllm_served_name,
            n_examples,
            prompt_type=pt,
            ravens_prompt_mode=ravens_prompt_mode,
        )
        summary = _make_model_summary(
            results_path,
            model_id=model_id,
            prompt_type=pt,
            n_examples=n_examples,
            gpu_tier=gpu_tier,
            vllm_served_name=vllm_served_name,
        )
        print(json.dumps(summary, indent=2), flush=True)
        hf_cache_volume.commit()
        return summary

    serve_cmd = _build_vllm_serve_cmd(model_id, gpu_tier)
    log_path = _vllm_log_path(model_id)
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
            )
            summary = _make_model_summary(
                results_path,
                model_id=model_id,
                prompt_type=pt,
                n_examples=n_examples,
                gpu_tier=gpu_tier,
                vllm_served_name=vllm_served_name,
            )
            summaries[pt] = summary
            print(json.dumps(summary, indent=2), flush=True)
        hf_cache_volume.commit()
        return summaries
    finally:
        _stop_vllm_server(proc, log_file, reader)


def _log_experiment_local(
    *,
    run_label: str,
    model_summaries: dict[str, dict[str, Any]],
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
    accuracy_delta: Optional[float] = None,
) -> None:
    """Append results to ``experiments.md`` on the local machine (after ``.remote()``)."""
    from experiment_log import append_experiment_entry

    settings_note = (
        f"{EXPERIMENT_SETTINGS_NOTE}; `--n-examples {n_examples}`; "
        f"`--prompt-type {prompt_type}`"
        + (
            f"; `--ravens-prompt-mode {resolve_instruction_prompt_mode(ravens_prompt_mode)}`"
            if prompt_type == "instruction"
            else ""
        )
    )
    append_experiment_entry(
        EXPERIMENTS_MD_LOCAL,
        run_label=run_label,
        model_summaries=model_summaries,
        max_tasks=max_tasks,
        accuracy_delta=accuracy_delta,
        settings_note=settings_note,
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
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any]:
    """Evaluate one HF model on ravens_numerical via in-container vLLM (T4)."""
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
def run_ravens_eval_a10g(
    model_id: str,
    max_tasks: Optional[int] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any]:
    """Evaluate one HF model on ravens_numerical via in-container vLLM (A10G)."""
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="A10G",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
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
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any]:
    """Evaluate one HF model on ravens_numerical via in-container vLLM (A100).

    Used for ≥10B models (e.g. Pythia-12B, Qwen3-14B) that OOM on A10G in fp16.
    """
    return _run_model_eval(
        model_id,
        max_tasks,
        gpu_tier="A100",
        n_examples=n_examples,
        prompt_type=prompt_type,
        ravens_prompt_mode=ravens_prompt_mode,
    )


def _remote_run_ravens_eval(
    model_id: str,
    max_tasks: Optional[int],
    n_examples: int,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> dict[str, Any]:
    tier = gpu_tier_for_model(model_id)
    kwargs = {
        "model_id": model_id,
        "max_tasks": max_tasks,
        "n_examples": n_examples,
        "prompt_type": prompt_type,
        "ravens_prompt_mode": ravens_prompt_mode,
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
    max_tasks: Optional[int] = 120,
    models: str = "sweep",
    n_examples: int = DEFAULT_N_EXAMPLES,
    prompt_type: str = "instruction",
    ravens_prompt_mode: str = "auto",
) -> None:
    """Local entry: ``modal run baby_reasoning_eval/modal_eval.py --models sweep``.

    ``--models`` accepts a HuggingFace id, comma-separated ids, or aliases:
    ``sweep`` (full scaling ladder), ``pythia``, ``qwen3``.

    ``--prompt-type`` is ``instruction`` (letter MCQ; default), ``completion``
    (bracket fill-in, baby-reasoning / matrix_easy style), or ``both`` (runs each
    sequentially in one vLLM session per model).

    Appends results to ``baby_reasoning_eval/experiments.md`` after each model.
    """
    if prompt_type not in PROMPT_TYPE_CLI:
        raise ValueError(
            f"--prompt-type must be one of {PROMPT_TYPE_CLI}, got {prompt_type!r}"
        )

    model_ids = resolve_models_arg(models)
    if not model_ids:
        raise ValueError(f"No models resolved from --models {models!r}")

    all_results: dict[str, Any] = {}
    for model_id in model_ids:
        print(
            f"\n=== Evaluating {model_id} (GPU {gpu_tier_for_model(model_id)}, "
            f"prompt={prompt_type}) ===\n",
            flush=True,
        )
        summary = _remote_run_ravens_eval(
            model_id,
            max_tasks,
            n_examples,
            prompt_type=prompt_type,
            ravens_prompt_mode=ravens_prompt_mode,
        )
        safe_key = model_id.replace("/", "--")

        if prompt_type == "both":
            assert isinstance(summary, dict)
            all_results[model_id] = summary
            for pt, pt_summary in summary.items():
                _log_experiment_local(
                    run_label=f"run_ravens_eval ({safe_key}, {pt})",
                    model_summaries={safe_key: pt_summary},
                    max_tasks=max_tasks,
                    n_examples=n_examples,
                    prompt_type=pt,
                    ravens_prompt_mode=ravens_prompt_mode,
                )
        else:
            assert isinstance(summary, dict) and "prompt_type" in summary
            all_results[model_id] = summary
            _log_experiment_local(
                run_label=f"run_ravens_eval ({safe_key}, {prompt_type})",
                model_summaries={safe_key: summary},
                max_tasks=max_tasks,
                n_examples=n_examples,
                prompt_type=prompt_type,
                ravens_prompt_mode=ravens_prompt_mode,
            )

    output = {
        "models": model_ids,
        "n_examples": n_examples,
        "prompt_type": prompt_type,
        "ravens_prompt_mode": ravens_prompt_mode,
        "max_tasks": max_tasks,
        "results": all_results,
    }
    print(json.dumps(output, indent=2))
