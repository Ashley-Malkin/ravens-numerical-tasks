"""Raven's numerical tasks from ``data/complete.json`` (or Webb ``tasks_webb.json``)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

from ravens_numerical.eval.tasks.base import ModelResponse, Stimulus, Task
from ravens_numerical.parsing.answer_parse import parse_answer
from ravens_numerical.paths import REPO_ROOT, COMPLETE_JSON
from ravens_numerical.prompts.prompts import (
    IclExample,
    build_prompt,
    completion_perm_invariant,
    expected_completion_answer,
    format_completion_answer,
)

PromptType = Literal["instruction", "completion"]
PromptMode = Literal["plain", "choice_only", "cot_choice"]


def _normalize_icl_bank(raw: Any) -> dict[str, list[IclExample]]:
    """Normalize an ``icl`` JSON object into a typed bank."""
    if not isinstance(raw, dict):
        raise ValueError('ICL bank must be a dict of task_type -> list of examples')
    bank: dict[str, list[IclExample]] = {}
    for task_type, examples in raw.items():
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"ICL bank for {task_type!r} must be a non-empty list")
        normalized: list[IclExample] = []
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                raise ValueError(f"ICL example {task_type}[{i}] must be a dict")
            if "matrix" not in ex or "answer_options" not in ex:
                raise ValueError(
                    f"ICL example {task_type}[{i}] needs matrix and answer_options"
                )
            item = dict(ex)
            if "correct_letter" not in item:
                if "correct_index" not in item:
                    raise ValueError(
                        f"ICL example {task_type}[{i}] needs correct_letter or correct_index"
                    )
                item["correct_letter"] = "ABCD"[int(item["correct_index"])]
            if "correct_index" not in item:
                item["correct_index"] = ord(str(item["correct_letter"]).upper()[0]) - ord(
                    "A"
                )
            if "task_type" not in item:
                item["task_type"] = task_type
            normalized.append(item)
        bank[str(task_type)] = normalized
    return bank


class RavensNumericalTask(Task):
    """Loads tasks from JSON; prompts via ``ravens_numerical.prompts``."""

    def __init__(
        self,
        tasks_json: Path | None = None,
        ravens_repo_root: Path | None = None,
        max_tasks: int | None = None,
        rng: random.Random | None = None,
        prompt_type: PromptType = "instruction",
        prompt_mode: PromptMode = "choice_only",
        icl_examples: dict[str, list[IclExample]] | None = None,
        load_icl_from_json: bool = False,
    ) -> None:
        self._repo_root = Path(ravens_repo_root) if ravens_repo_root is not None else REPO_ROOT
        self._tasks_path = Path(tasks_json) if tasks_json is not None else COMPLETE_JSON
        self._max_tasks = max_tasks
        self._rng = rng or random.Random()
        self._prompt_type: PromptType = prompt_type
        self._prompt_mode: PromptMode = prompt_mode
        self._tasks: list[dict[str, Any]] = []
        self._icl_examples: dict[str, list[IclExample]] | None = icl_examples
        self._load_icl_from_json = load_icl_from_json
        self._load_tasks()

    def _load_tasks(self) -> None:
        if not self._tasks_path.is_file():
            raise FileNotFoundError(
                f"ravens tasks JSON not found: {self._tasks_path}. "
                "Set --ravens-tasks-json or install the package with data/complete.json."
            )
        data = json.loads(self._tasks_path.read_text(encoding="utf-8"))
        raw: list = data.get("tasks", data)
        if not isinstance(raw, list):
            raise ValueError('tasks.json must contain a list or {"tasks": [...]}')
        if self._max_tasks is not None:
            raw = raw[: self._max_tasks]
        self._tasks = [
            t for t in raw if isinstance(t, dict) and "matrix" in t and "answer_options" in t
        ]
        if not self._tasks:
            raise ValueError(f"No valid tasks loaded from {self._tasks_path}")

        if self._icl_examples is None and self._load_icl_from_json and "icl" in data:
            # Top-level type→demos bank (Webb / legacy shared-bank ICL files).
            self._icl_examples = _normalize_icl_bank(data["icl"])

    def _task_to_stimulus(self, task: dict[str, Any]) -> Stimulus:
        if self._prompt_type == "completion":
            perm = task.get("perm_invariant")
            return Stimulus(
                query="",
                expected=expected_completion_answer(task),
                metadata={
                    "task": task,
                    "perm_invariant": completion_perm_invariant(
                        task["task_type"],
                        perm_invariant=perm if isinstance(perm, bool) else None,
                    ),
                },
                answer_choices=[format_completion_answer(opt) for opt in task["answer_options"]],
            )

        letter = task.get("correct_letter")
        if not letter:
            letter = "ABCD"[int(task["correct_index"])]
        return Stimulus(
            query="",
            expected=str(letter).strip().upper()[:1],
            metadata={"task": task},
        )

    def canonical_stimuli(self) -> list[Stimulus]:
        return [self._task_to_stimulus(t) for t in self._tasks]

    def generate_stimulus(self, n_examples: int = 3) -> Stimulus:
        _ = n_examples
        return self._task_to_stimulus(self._rng.choice(self._tasks))

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        task = stimulus.metadata["task"]
        mode = self._prompt_mode if self._prompt_type == "instruction" else "plain"
        # Prefer per-task ``icl`` (unique demos per item) over a shared bank.
        icl_examples = self._icl_examples
        per_task = task.get("icl")
        if isinstance(per_task, list) and per_task:
            icl_examples = {str(task["task_type"]): list(per_task)}
        return build_prompt(
            task,
            mode=mode,
            n_examples=n_examples,
            prompt_type=self._prompt_type,
            icl_examples=icl_examples,
        )

    @property
    def uses_choice_only_metrics(self) -> bool:
        return self._prompt_type == "instruction" and self._prompt_mode == "choice_only"

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        if self._prompt_type == "completion":
            text = response.text.split("]")[0].strip()
            expected = stimulus.expected.strip()
            if stimulus.metadata.get("perm_invariant", False):
                return set(text.split()) == set(expected.split())
            return text == expected

        task = stimulus.metadata["task"]
        ci = int(task["correct_index"])
        pred = parse_answer(
            response.text,
            answer_options=task["answer_options"],
            correct_index=ci,
        )
        return pred == ci

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        if self._prompt_type == "completion":
            return choice + "]"
        return choice
