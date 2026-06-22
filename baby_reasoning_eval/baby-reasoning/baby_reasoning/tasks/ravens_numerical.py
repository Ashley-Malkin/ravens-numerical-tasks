"""Raven's numerical tasks from ravens-numerical-tasks ``tasks.json`` (plain letter prompts)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from baby_reasoning.tasks.base import ModelResponse, Stimulus, Task


def _default_ravens_repo_root() -> Path:
    """Parent of ``baby_reasoning_eval/`` when this package lives under ``baby_reasoning_eval/baby-reasoning``."""
    # baby_reasoning/tasks/ravens_numerical.py -> tasks, baby_reasoning, clone root, baby_reasoning_eval, ravens root
    return Path(__file__).resolve().parents[4]


class RavensNumericalTask(Task):
    """Loads tasks from JSON; prompts match ``evaluate.py`` plain mode via ``ravens_prompts.build_prompt``."""

    def __init__(
        self,
        tasks_json: Path | None = None,
        ravens_repo_root: Path | None = None,
        max_tasks: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._repo_root = Path(ravens_repo_root) if ravens_repo_root is not None else _default_ravens_repo_root()
        self._tasks_path = Path(tasks_json) if tasks_json is not None else self._repo_root / "tasks.json"
        self._max_tasks = max_tasks
        self._rng = rng or random.Random()
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
        return build_prompt(task, mode="plain", n_examples=n_examples)

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
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
