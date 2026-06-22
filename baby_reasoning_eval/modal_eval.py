#!/usr/bin/env python3
"""Run ravens_numerical eval on Modal (Option A): vLLM server inside the GPU container.

Compare Pythia vs Qwen3 on the same ``tasks.json`` / prompts as ``baby_reasoning_eval`` locally,
using ``script/run --backend vllm`` against ``http://127.0.0.1:8000``.

Setup (once on your machine):

    pip install modal
    modal setup

Smoke test:

    modal run baby_reasoning_eval/modal_eval.py --max-tasks 10 --model both

Full eval (120 tasks; appends to ``experiments.md``):

    modal run baby_reasoning_eval/modal_eval.py --model both

Direct remote functions (``::run_pythia_ravens``, etc.) skip ``experiments.md`` logging; use the entrypoint above.
"""

from __future__ import annotations

import json
import os
import subprocess
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

# Paths inside the Modal container.
CONTAINER_RAVENS_ROOT = "/root/ravens"
CONTAINER_BABY_ROOT = f"{CONTAINER_RAVENS_ROOT}/baby_reasoning_eval/baby-reasoning"
CONTAINER_TASKS_JSON = f"{CONTAINER_RAVENS_ROOT}/tasks.json"

EXPERIMENT_SETTINGS_NOTE = (
    "Modal vLLM (`modal_eval.py`); `--max-model-len 2048`; `--attention-backend FLASH_ATTN`"
)

DEFAULT_N_EXAMPLES = 0

DEFAULT_PYTHIA_VLLM = "EleutherAI/pythia-70m-deduped"
# HF id closest to local Ollama ``qwen3:8b`` for vLLM comparison.
DEFAULT_QWEN3_VLLM = "Qwen/Qwen3-8B"

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


def _run_script(
    model_id: str,
    max_tasks: Optional[int],
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> Path:
    """Start vLLM, run baby-reasoning CLI, return path to results JSON."""
    served = vllm_served_name or model_id
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CONTAINER_RAVENS_ROOT}:{CONTAINER_BABY_ROOT}"
    # debian_slim has no nvcc; FlashInfer sampling JIT would fail without this.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    serve_cmd = [
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
        "FLASH_ATTN",
        "--max-model-len",
        "2048",
    ]
    log_path = _vllm_log_path(model_id)
    proc, reader, log_file = _start_vllm_server(serve_cmd, log_path, env)
    try:
        _wait_for_vllm(proc, log_path, serve_cmd)

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
        ]
        if max_tasks is not None:
            cmd.extend(["--ravens-max-tasks", str(max_tasks)])

        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=CONTAINER_BABY_ROOT, env=env)

        results_root = Path(CONTAINER_BABY_ROOT) / "results"
        results_glob = f"**/ravens_numerical/{n_examples}_examples.json"
        candidates = sorted(
            results_root.glob(results_glob),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No results under {results_root}; expected ravens_numerical/{n_examples}_examples.json"
            )
        return candidates[-1]
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


def _run_model_eval(
    model_id: str,
    max_tasks: Optional[int],
    vllm_served_name: Optional[str] = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    results_path = _run_script(model_id, max_tasks, vllm_served_name, n_examples)
    summary = _summarize_results(results_path)
    summary["model_id"] = model_id
    summary["vllm_served_name"] = vllm_served_name or model_id
    summary["n_examples"] = n_examples
    print(json.dumps(summary, indent=2), flush=True)
    hf_cache_volume.commit()
    return summary


def _log_experiment_local(
    *,
    run_label: str,
    result: dict[str, Any],
    max_tasks: Optional[int],
    n_examples: int,
) -> None:
    """Append results to ``experiments.md`` on the local machine (after ``.remote()``)."""
    from experiment_log import append_experiment_entry

    settings_note = f"{EXPERIMENT_SETTINGS_NOTE}; `--n-examples {n_examples}`"
    if "pythia" in result and "qwen3" in result:
        append_experiment_entry(
            EXPERIMENTS_MD_LOCAL,
            run_label=run_label,
            model_summaries={"pythia": result["pythia"], "qwen3": result["qwen3"]},
            max_tasks=max_tasks,
            accuracy_delta=result.get("accuracy_delta_qwen3_minus_pythia"),
            settings_note=settings_note,
        )
    else:
        key = "pythia" if "pythia" in run_label else "qwen3"
        append_experiment_entry(
            EXPERIMENTS_MD_LOCAL,
            run_label=run_label,
            model_summaries={key: result},
            max_tasks=max_tasks,
            settings_note=settings_note,
        )


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
    return _run_model_eval(model_id, max_tasks, n_examples=n_examples)


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
    return _run_model_eval(model_id, max_tasks, n_examples=n_examples)


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
    pythia = _run_model_eval(pythia_model_id, max_tasks, n_examples=n_examples)
    qwen3 = _run_model_eval(qwen3_model_id, max_tasks, n_examples=n_examples)
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
    model: str = "both",
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> None:
    """Local entry: ``modal run baby_reasoning_eval/modal_eval.py --model both`` (120 tasks).

    Appends results to ``baby_reasoning_eval/experiments.md`` on this machine after each run.
    """
    if model == "pythia":
        result = run_pythia_ravens.remote(max_tasks=max_tasks, n_examples=n_examples)
        _log_experiment_local(
            run_label="run_pythia_ravens",
            result=result,
            max_tasks=max_tasks,
            n_examples=n_examples,
        )
    elif model == "qwen3":
        result = run_qwen3_ravens.remote(max_tasks=max_tasks, n_examples=n_examples)
        _log_experiment_local(
            run_label="run_qwen3_ravens",
            result=result,
            max_tasks=max_tasks,
            n_examples=n_examples,
        )
    else:
        result = run_compare_ravens.remote(max_tasks=max_tasks, n_examples=n_examples)
        _log_experiment_local(
            run_label="run_compare_ravens",
            result=result,
            max_tasks=max_tasks,
            n_examples=n_examples,
        )
    print(json.dumps(result, indent=2))
