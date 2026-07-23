from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro

from ravens_numerical.eval.backends import (
    ForcedChoiceVLLMBackend,
    OllamaBackend,
    Olmo2ChoiceOnlyVLLMBackend,
    Olmo2InstructChoiceOnlyVLLMBackend,
    PythiaChoiceOnlyVLLMBackend,
    Qwen3ChoiceOnlyVLLMBackend,
    RobertaMLMBackend,
    VLLMBackend,
)
from ravens_numerical.eval.runner import evaluate, save_results
from ravens_numerical.eval.tasks.base import ModelBackend, Task
from ravens_numerical.eval.tasks.legacy.hierarchical import HierarchicalTask
from ravens_numerical.eval.tasks.legacy.matrix import MatrixTask
from ravens_numerical.eval.tasks.legacy.matrix_easy import MatrixEasyTask
from ravens_numerical.eval.tasks.ravens import RavensNumericalTask
from ravens_numerical.eval.tasks.legacy.rules import RulesTask
from ravens_numerical.models.registry import (
    base_model_id,
    is_miniberta_model,
    is_olmo2_instruct_model,
    is_olmo2_model,
    is_pythia_checkpoint_model_id,
    is_qwen3_instruct_model,
    resolve_instruction_prompt_mode,
)
from ravens_numerical.paths import REPO_ROOT, RUNS_DIR, TASKS_ABA_JSON, TASKS_JSON, TASKS_WEBB_JSON

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

SuiteName = Literal["ravens", "webb", "aba", "hierarchical"]

BackendName = Literal["vllm", "ollama", "hf"]
InstructionPromptMode = Literal["auto", "plain", "choice_only", "cot_choice"]
ScoreMode = Literal["auto", "free_gen", "forced_choice"]

_SUITE_TASKS: dict[SuiteName, list[TaskName]] = {
    "ravens": ["ravens_numerical"],
    "webb": ["ravens_numerical"],
    "aba": ["rules"],
    "hierarchical": ["hierarchical"],
}

_BABY_N_EXAMPLES = (0, 5, 10, 15, 20)
_RAVENS_N_EXAMPLES_DEFAULT = (0, 3)
_BABY_SUITES = frozenset({"aba", "hierarchical"})


def _ravens_repo_root(cfg: "Config") -> Path:
    if cfg.ravens_repo_root is not None:
        return Path(cfg.ravens_repo_root)
    return REPO_ROOT


def resolve_tasks(cfg: "Config") -> list[TaskName]:
    """Return the task list: explicit ``--tasks`` wins, else derive from ``task_type``."""
    if cfg.tasks:
        return list(cfg.tasks)
    return list(_SUITE_TASKS[cfg.task_type])


def resolve_n_examples(cfg: "Config") -> list[int]:
    """Return n_examples list; empty config uses suite defaults."""
    if cfg.n_examples:
        values = list(cfg.n_examples)
    elif cfg.task_type in _BABY_SUITES:
        values = list(_BABY_N_EXAMPLES)
    else:
        values = list(_RAVENS_N_EXAMPLES_DEFAULT)

    if cfg.task_type in _BABY_SUITES:
        bad = [n for n in values if n not in _BABY_N_EXAMPLES]
        if bad:
            raise ValueError(
                f"--task-type {cfg.task_type} only supports --n-examples in "
                f"{_BABY_N_EXAMPLES}; got {bad}"
            )
    return values


def resolve_score_mode(cfg: "Config", model_name: str) -> Literal["free_gen", "forced_choice"]:
    """Expand ``auto``: Pythia ``@stepN`` checkpoints → forced_choice; else free_gen."""
    if cfg.score_mode == "forced_choice":
        return "forced_choice"
    if cfg.score_mode == "free_gen":
        return "free_gen"
    if is_pythia_checkpoint_model_id(model_name):
        return "forced_choice"
    return "free_gen"


def resolve_ravens_prompt_type(
    cfg: "Config",
    score_mode: Literal["free_gen", "forced_choice"],
) -> Literal["instruction", "completion"]:
    """Resolve Raven/Webb prompt type; ``forced_choice`` always uses completion."""
    if score_mode == "forced_choice":
        return "completion"
    return cfg.ravens_prompt_type


@dataclass
class Config:
    """Run abstract rule learning evaluations against vLLM or Ollama models."""

    models: list[str]
    """Model ids: Hugging Face ids for vLLM (e.g. EleutherAI/pythia-70m-deduped), Ollama names for ``ollama`` (e.g. qwen3:8b)."""

    backend: BackendName = "vllm"
    """``vllm``: OpenAI-compatible completions API. ``ollama``: ``/api/generate``. ``hf``: in-process HuggingFace (MiniBERTa)."""

    hf_device: str = "cuda"
    """Device for ``hf`` backend (``cuda`` or ``cpu``)."""

    task_type: SuiteName = "ravens"
    """Suite: ``ravens`` / ``webb`` → ``ravens_numerical``; ``aba`` → rules; ``hierarchical`` → hierarchical equality."""

    tasks: list[TaskName] = field(default_factory=list)
    """Tasks to run. When empty (default), derived from ``task_type``. Explicit values override the suite."""

    n_examples: list[int] = field(default_factory=list)
    """In-context example counts. Empty (default) → ``[0, 3]`` for ravens/webb, ``[0, 5, 10, 15, 20]`` for aba/hierarchical.

    For ``ravens``, demos come from ``ravens_prompts.IN_CONTEXT_EXAMPLES``.
    For ``webb``, demos come from the ``icl`` section of ``tasks_webb.json``.
    For aba / hierarchical, demos come from shared pools in ``tasks_aba.json``.
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
    """Path to tasks JSON for ``ravens_numerical``. Defaults to ``data/tasks.json`` (``ravens``) or ``data/tasks_webb.json`` (``webb``)."""

    aba_tasks_json: Path | None = None
    """Path to ``tasks_aba.json`` for aba / hierarchical. Defaults to ``data/tasks_aba.json``."""

    ravens_repo_root: Path | None = None
    """Root of ravens-numerical-tasks repo. Resolved automatically when unset."""

    ravens_max_tasks: int | None = None
    """If set, only the first N tasks from the JSON are evaluated."""

    ravens_prompt_type: Literal["instruction", "completion"] = "completion"
    """``completion``: bracket fill-in (default). ``instruction``: letter MCQ.

    With ``score_mode=forced_choice``, completion is always used regardless of this flag.
    """

    ravens_prompt_mode: InstructionPromptMode = "auto"
    """Instruction-only output mode. ``auto`` → ``choice_only`` (JSON ``{"choice":"B"}``). Ignored for completion."""

    score_mode: ScoreMode = "forced_choice"
    """How ``correct`` is decided for tasks with ``answer_choices``.

    ``forced_choice`` (default): argmax of completion logprobs among choices
    (hierarchical also constrains generation via vLLM structured choice).
    ``free_gen``: parse free generation. ``auto`` → ``forced_choice`` for Pythia
    ``@stepN`` checkpoints, else ``free_gen``. Raven's instruction ``choice_only``
    is unchanged when using ``--ravens-prompt-type instruction --score-mode free_gen``.
    """


def _resolved_instruction_prompt_mode(cfg: Config) -> str:
    return resolve_instruction_prompt_mode(cfg.ravens_prompt_mode)


def _aba_tasks_json(cfg: Config) -> Path:
    return cfg.aba_tasks_json or TASKS_ABA_JSON


def _default_ravens_tasks_json(cfg: Config) -> Path:
    if cfg.task_type == "webb":
        return TASKS_WEBB_JSON
    return TASKS_JSON


def _instantiate_task(
    task_name: TaskName,
    cfg: Config,
    score_mode: Literal["free_gen", "forced_choice"] = "free_gen",
) -> Task:
    if task_name == "ravens_numerical":
        prompt_type = resolve_ravens_prompt_type(cfg, score_mode)
        prompt_mode = _resolved_instruction_prompt_mode(cfg)
        if prompt_mode not in ("plain", "choice_only", "cot_choice"):
            raise ValueError(f"Unsupported ravens_prompt_mode: {prompt_mode!r}")
        tasks_json = cfg.ravens_tasks_json or _default_ravens_tasks_json(cfg)
        return RavensNumericalTask(
            tasks_json=tasks_json,
            ravens_repo_root=cfg.ravens_repo_root or _ravens_repo_root(cfg),
            max_tasks=cfg.ravens_max_tasks,
            prompt_type=prompt_type,
            prompt_mode=prompt_mode,  # type: ignore[arg-type]
            load_icl_from_json=cfg.task_type == "webb",
        )
    if task_name == "rules":
        return RulesTask(tasks_json=_aba_tasks_json(cfg))
    if task_name == "hierarchical":
        return HierarchicalTask(tasks_json=_aba_tasks_json(cfg))
    ctor = TASK_MAP[task_name]
    return ctor()


def _make_backend(
    cfg: Config,
    model_name: str,
    task: Task,
    score_mode: Literal["free_gen", "forced_choice"] = "free_gen",
) -> ModelBackend:
    hf_model = base_model_id(model_name)
    if cfg.backend == "ollama":
        return OllamaBackend(
            model_name,
            base_url=cfg.ollama_base_url,
            timeout=cfg.ollama_timeout,
            max_tokens=cfg.ollama_max_tokens,
        )

    if cfg.backend == "hf" or is_miniberta_model(hf_model):
        # Raven's instruction modes use letter/JSON PLL; baby suites and other
        # answer-choice tasks use completion-span PLL over the options.
        if getattr(task, "uses_choice_only_metrics", False):
            prompt_type = cfg.ravens_prompt_type
            prompt_mode = _resolved_instruction_prompt_mode(cfg)
        elif isinstance(task, RavensNumericalTask):
            prompt_type = cfg.ravens_prompt_type
            prompt_mode = (
                _resolved_instruction_prompt_mode(cfg)
                if cfg.ravens_prompt_type == "instruction"
                else "plain"
            )
        else:
            prompt_type = "completion"
            prompt_mode = "plain"
        return RobertaMLMBackend(
            hf_model,
            device=cfg.hf_device,
            prompt_type=prompt_type,
            prompt_mode=prompt_mode,
        )

    vllm_model = hf_model
    if score_mode == "forced_choice" and isinstance(task, RavensNumericalTask):
        return VLLMBackend(vllm_model, cfg.base_url)

    if (
        cfg.ravens_prompt_type == "instruction"
        and getattr(task, "uses_choice_only_metrics", False)
    ):
        if is_qwen3_instruct_model(hf_model):
            return Qwen3ChoiceOnlyVLLMBackend(vllm_model, cfg.base_url)
        if is_olmo2_instruct_model(hf_model):
            return Olmo2InstructChoiceOnlyVLLMBackend(vllm_model, cfg.base_url)
        if is_olmo2_model(hf_model):
            return Olmo2ChoiceOnlyVLLMBackend(vllm_model, cfg.base_url)
        # Pythia, BabyLM, Qwen3-Base, and other completions + guided JSON models.
        return PythiaChoiceOnlyVLLMBackend(vllm_model, cfg.base_url)

    if score_mode == "forced_choice":
        # Raven/ABA/matrix strings like ``"12]"`` / ``" pe"`` can 500 under vLLM
        # structured choice. ``forced_choice`` ``correct`` comes from echo logprob
        # argmax in ``evaluate``; constrain generation only for hierarchical digits.
        if isinstance(task, HierarchicalTask):
            return ForcedChoiceVLLMBackend(vllm_model, cfg.base_url)
        return VLLMBackend(vllm_model, cfg.base_url)

    return VLLMBackend(vllm_model, cfg.base_url)


def _systematic_kwargs(task: Task, n_stimuli: int, n_examples: int) -> dict:
    """Return keyword arguments for systematic_stimuli based on task type."""
    if isinstance(task, RulesTask):
        return {"n_per_rule": n_stimuli, "n_examples": n_examples}
    if isinstance(task, HierarchicalTask):
        return {"n_per_pattern": n_stimuli, "n_examples": n_examples}
    raise TypeError(f"No systematic generation for {type(task).__name__}")


def _results_suffix(
    cfg: Config,
    task_name: TaskName,
    score_mode: Literal["free_gen", "forced_choice"] = "free_gen",
) -> str | None:
    if task_name == "ravens_numerical":
        if score_mode == "forced_choice" or cfg.ravens_prompt_type == "completion":
            return "completion"
        if _resolved_instruction_prompt_mode(cfg) == "choice_only":
            return "choice_only"
        return None
    if score_mode == "forced_choice" and task_name in ("rules", "hierarchical", "matrix", "matrix_easy"):
        return "forced_choice"
    return None


def run(cfg: Config) -> None:
    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    n_examples = resolve_n_examples(cfg)
    max_n_examples = max(n_examples)
    tasks = resolve_tasks(cfg)

    for model_name in cfg.models:
        score_mode = resolve_score_mode(cfg, model_name)
        for task_name in tasks:
            task = _instantiate_task(task_name, cfg, score_mode=score_mode)
            backend = _make_backend(cfg, model_name, task, score_mode=score_mode)
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
            for n_ex in n_examples:
                print(
                    f"{model_name}  {task_name}/{n_ex}_examples ... ",
                    end="",
                    flush=True,
                )
                results = evaluate(
                    task, backend, n_ex, stimuli, score_mode=score_mode
                )
                path = save_results(
                    results,
                    model_name,
                    task_name,
                    n_ex,
                    results_dir=cfg.results_dir or RUNS_DIR,
                    run_id=run_id,
                    results_suffix=_results_suffix(cfg, task_name, score_mode),
                )
                n_correct = sum(r.score.correct for r in results)
                print(f"{n_correct}/{len(results)} correct → {path}")


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
