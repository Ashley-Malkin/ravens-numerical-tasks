"""Raven's numerical tasks from ravens-numerical-tasks ``tasks.json``."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Literal

from baby_reasoning.tasks.base import ModelResponse, Stimulus, Task

PromptType = Literal["instruction", "completion"]
PromptMode = Literal["plain", "choice_only", "cot_choice"]


def _default_ravens_repo_root() -> Path:
    """Parent of ``baby_reasoning_eval/`` when this package lives under ``baby_reasoning_eval/baby-reasoning``."""
    # baby_reasoning/tasks/ravens_numerical.py -> tasks, baby_reasoning, clone root, baby_reasoning_eval, ravens root
    return Path(__file__).resolve().parents[4]


class RavensNumericalTask(Task):
    """Loads tasks from JSON; prompts via ``ravens_prompts.build_prompt``."""

    def __init__(
        self,
        tasks_json: Path | None = None,
        ravens_repo_root: Path | None = None,
        max_tasks: int | None = None,
        rng: random.Random | None = None,
        prompt_type: PromptType = "instruction",
        prompt_mode: PromptMode = "choice_only",
    ) -> None:
        self._repo_root = Path(ravens_repo_root) if ravens_repo_root is not None else _default_ravens_repo_root()
        self._tasks_path = Path(tasks_json) if tasks_json is not None else self._repo_root / "tasks.json"
        self._max_tasks = max_tasks
        self._rng = rng or random.Random()
        self._prompt_type: PromptType = prompt_type
        self._prompt_mode: PromptMode = prompt_mode
        self._tasks: list[dict[str, Any]] = []
        self._load_tasks()

    def _ensure_import_path(self) -> None:
        root = str(self._repo_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    def _load_tasks(self) -> None:
        if not self._tasks_path.is_file():
            raise FileNotFoundError(
                f"ravens tasks JSON not found: {self._tasks_path}. "
                "Set --ravens-tasks-json or --ravens-repo-root."
            )
        data = json.loads(self._tasks_path.read_text(encoding="utf-8"))
        raw: list = data.get("tasks", data)
        if not isinstance(raw, list):
            raise ValueError("tasks.json must contain a list or {\"tasks\": [...]}")
        if self._max_tasks is not None:
            raw = raw[: self._max_tasks]
        self._tasks = [t for t in raw if isinstance(t, dict) and "matrix" in t and "answer_options" in t]
        if not self._tasks:
            raise ValueError(f"No valid tasks loaded from {self._tasks_path}")

    def _task_to_stimulus(self, task: dict[str, Any]) -> Stimulus:
        if self._prompt_type == "completion":
            self._ensure_import_path()
            from ravens_prompts import (
                completion_perm_invariant,
                expected_completion_answer,
                format_completion_answer,
            )

            return Stimulus(
                query="",
                expected=expected_completion_answer(task),
                metadata={
                    "task": task,
                    "perm_invariant": completion_perm_invariant(task["task_type"]),
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
        self._ensure_import_path()
        from ravens_prompts import build_prompt

        task = stimulus.metadata["task"]
        mode = self._prompt_mode if self._prompt_type == "instruction" else "plain"
        return build_prompt(
            task,
            mode=mode,
            n_examples=n_examples,
            prompt_type=self._prompt_type,
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

        self._ensure_import_path()
        from ravens_answer_parse import parse_answer

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
