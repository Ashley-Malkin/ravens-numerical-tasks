from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from baby_reasoning.model import (
    OllamaBackend,
    PythiaChoiceOnlyVLLMBackend,
    Qwen3ChoiceOnlyVLLMBackend,
    VLLMBackend,
)
from baby_reasoning.runner import evaluate, save_results
from baby_reasoning.tasks.base import ModelBackend, Task
from baby_reasoning.tasks.hierarchical import HierarchicalTask
from baby_reasoning.tasks.matrix import MatrixTask
from baby_reasoning.tasks.matrix_easy import MatrixEasyTask
from baby_reasoning.tasks.ravens_numerical import RavensNumericalTask
from baby_reasoning.tasks.rules import RulesTask

TASK_MAP = {
    "rules": RulesTask,
    "hierarchical": HierarchicalTask,
    "matrix": MatrixTask,
    "matrix_easy": MatrixEasyTask,
}

TaskName = Literal[
    "rules",
    "hierarchical",
    "matrix",
    "matrix_easy",
    "ravens_numerical",
]


BackendName = Literal["vllm", "ollama"]
InstructionPromptMode = Literal["auto", "plain", "choice_only", "cot_choice"]


def _ravens_repo_root(cfg: "Config") -> Path:
    if cfg.ravens_repo_root is not None:
        return Path(cfg.ravens_repo_root)
    return Path(__file__).resolve().parents[4]


def _ensure_repo_on_path(repo_root: Path) -> None:
    root_s = str(repo_root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


@dataclass
class Config:
    """Run abstract rule learning evaluations against vLLM or Ollama models."""

    models: list[str]
    """Model ids: Hugging Face ids for vLLM (e.g. EleutherAI/pythia-70m-deduped), Ollama names for ``ollama`` (e.g. qwen3:8b)."""

    backend: BackendName = "vllm"
    """``vllm``: OpenAI-compatible completions API. ``ollama``: ``/api/generate`` (Qwen3, etc.)."""

    tasks: list[TaskName] = field(
        default_factory=lambda: ["rules", "hierarchical", "matrix"]
    )
    """Tasks to run. Defaults to all three."""

    n_examples: list[int] = field(default_factory=lambda: [0, 3])
    """Number of in-context examples to evaluate. Defaults to [0, 3].

    For ``ravens_numerical``, this controls how many demos from ``ravens_prompts.IN_CONTEXT_EXAMPLES``
    are prepended (``0`` = none, ``3`` = all three per task type).
    """

    base_url: str = "http://localhost:8000"
    """vLLM server base URL (used when ``backend`` is ``vllm``)."""

    ollama_base_url: str = "http://localhost:11434"
    """Ollama server base URL (used when ``backend`` is ``ollama``)."""

    ollama_timeout: int = 300
    """HTTP timeout in seconds per request when ``backend`` is ``ollama``."""

    ollama_max_tokens: int = 64
    """Max generated tokens (``num_predict``) when ``backend`` is ``ollama``."""

    results_dir: Path | None = None
    """Override the default results directory."""

    n_stimuli: int | None = None
    """If set, generate N random stimuli instead of using the canonical set."""

    systematic: bool = False
    """Use systematic stimulus generation (balanced across rule/pattern types)."""

    ravens_tasks_json: Path | None = None
    """Path to ``tasks.json`` for ``ravens_numerical``. Defaults to ``<ravens_repo_root>/tasks.json``."""

    ravens_repo_root: Path | None = None
    """Root of ravens-numerical-tasks repo (contains ``ravens_prompts.py``). Resolved automatically when unset."""

    ravens_max_tasks: int | None = None
    """If set, only the first N tasks from the JSON are evaluated."""

    ravens_prompt_type: Literal["instruction", "completion"] = "instruction"
    """``instruction``: letter MCQ (default). ``completion``: bracket fill-in (baby-reasoning / matrix_easy style)."""

    ravens_prompt_mode: InstructionPromptMode = "auto"
    """Instruction-only output mode. ``auto`` → ``choice_only`` (JSON ``{"choice":"B"}``). Ignored for completion."""


def _resolved_instruction_prompt_mode(cfg: Config) -> str:
    _ensure_repo_on_path(_ravens_repo_root(cfg))
    from ravens_eval_models import resolve_instruction_prompt_mode

    return resolve_instruction_prompt_mode(cfg.ravens_prompt_mode)


def _instantiate_task(task_name: TaskName, cfg: Config) -> Task:
    if task_name == "ravens_numerical":
        prompt_mode = _resolved_instruction_prompt_mode(cfg)
        if prompt_mode not in ("plain", "choice_only", "cot_choice"):
            raise ValueError(f"Unsupported ravens_prompt_mode: {prompt_mode!r}")
        return RavensNumericalTask(
            tasks_json=cfg.ravens_tasks_json,
            ravens_repo_root=cfg.ravens_repo_root,
            max_tasks=cfg.ravens_max_tasks,
            prompt_type=cfg.ravens_prompt_type,
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
        )
    ctor = TASK_MAP[task_name]
    return ctor()


def _make_backend(cfg: Config, model_name: str, task: Task) -> ModelBackend:
    if cfg.backend == "ollama":
        return OllamaBackend(
            model_name,
            base_url=cfg.ollama_base_url,
            timeout=cfg.ollama_timeout,
            max_tokens=cfg.ollama_max_tokens,
        )

    if (
        cfg.ravens_prompt_type == "instruction"
        and getattr(task, "uses_choice_only_metrics", False)
    ):
        _ensure_repo_on_path(_ravens_repo_root(cfg))
        from ravens_eval_models import is_qwen3_model

        if is_qwen3_model(model_name):
            return Qwen3ChoiceOnlyVLLMBackend(model_name, cfg.base_url)
        return PythiaChoiceOnlyVLLMBackend(model_name, cfg.base_url)

    return VLLMBackend(model_name, cfg.base_url)


def _systematic_kwargs(task: Task, n_stimuli: int, n_examples: int) -> dict:
    """Return keyword arguments for systematic_stimuli based on task type."""
    if isinstance(task, RulesTask):
        return {"n_per_rule": n_stimuli, "n_examples": n_examples}
    if isinstance(task, HierarchicalTask):
        return {"n_per_pattern": n_stimuli, "n_examples": n_examples}
    raise TypeError(f"No systematic generation for {type(task).__name__}")


def _results_suffix(cfg: Config, task_name: TaskName) -> str | None:
    if task_name != "ravens_numerical":
        return None
    if cfg.ravens_prompt_type == "completion":
        return "completion"
    if _resolved_instruction_prompt_mode(cfg) == "choice_only":
        return "choice_only"
    return None


def run(cfg: Config) -> None:
    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    max_n_examples = max(cfg.n_examples)

    for model_name in cfg.models:
        for task_name in cfg.tasks:
            task = _instantiate_task(task_name, cfg)
            backend = _make_backend(cfg, model_name, task)
            # Generate stimuli once per model×task, reuse across n_examples values
            if cfg.n_stimuli is not None:
                if cfg.systematic and hasattr(task, "systematic_stimuli"):
                    stimuli = task.systematic_stimuli(
                        **_systematic_kwargs(task, cfg.n_stimuli, max_n_examples)
                    )
                elif hasattr(task, "all_stimuli"):
                    stimuli = task.all_stimuli()
                else:
                    stimuli = [
                        task.generate_stimulus(n_examples=max_n_examples)
                        for _ in range(cfg.n_stimuli)
                    ]
            else:
                stimuli = None
            for n_ex in cfg.n_examples:
                print(
                    f"{model_name}  {task_name}/{n_ex}_examples ... ",
                    end="",
                    flush=True,
                )
                results = evaluate(task, backend, n_ex, stimuli)
                path = save_results(
                    results,
                    model_name,
                    task_name,
                    n_ex,
                    results_dir=cfg.results_dir,
                    run_id=run_id,
                    results_suffix=_results_suffix(cfg, task_name),
                )
                n_correct = sum(r.score.correct for r in results)
                print(f"{n_correct}/{len(results)} correct → {path}")
