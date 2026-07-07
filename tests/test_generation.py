#!/usr/bin/env python3
"""Tests for tasks.json layout and max_tasks slicing."""

from __future__ import annotations

import json
from collections import Counter

from ravens_numerical.eval.tasks.ravens import RavensNumericalTask
from ravens_numerical.generation.generator import TASK_TYPE_CYCLE, interleave_tasks_by_type
from ravens_numerical.paths import TASKS_JSON


def _load_tasks() -> list[dict]:
    data = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data


def test_tasks_interleaved_cycle():
    tasks = _load_tasks()
    assert len(tasks) == 140
    counts = Counter(t["task_type"] for t in tasks)
    for tt in TASK_TYPE_CYCLE:
        assert counts[tt] == 20
    for i, task in enumerate(tasks):
        assert task["task_type"] == TASK_TYPE_CYCLE[i % len(TASK_TYPE_CYCLE)]


def test_interleave_idempotent():
    tasks = _load_tasks()
    assert interleave_tasks_by_type(tasks) == tasks


def test_max_tasks_includes_constancy_row():
    tasks = _load_tasks()
    first_10 = tasks[:10]
    types = {t["task_type"] for t in first_10}
    assert "constancy_row" in types


def test_ravens_numerical_max_tasks_includes_constancy_row():
    task = RavensNumericalTask(
        max_tasks=10,
        prompt_type="instruction",
    )
    types = {s.metadata["task"]["task_type"] for s in task.canonical_stimuli()}
    assert "constancy_row" in types
