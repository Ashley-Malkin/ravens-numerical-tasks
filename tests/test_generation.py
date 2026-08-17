#!/usr/bin/env python3
"""Tests for tasks.json / complete.json layout and max_tasks slicing."""

from __future__ import annotations

import json
from collections import Counter

from ravens_numerical.eval.tasks.ravens import RavensNumericalTask
from ravens_numerical.generation.generator import (
    CHALLENGE_TYPE_CYCLE,
    LEGACY_TASK_TYPE_CYCLE,
    TASK_TYPE_CYCLE,
    interleave_tasks_by_type,
)
from ravens_numerical.paths import COMPLETE_JSON, TASKS_JSON


def _load(path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tasks"] if isinstance(data, dict) else data


def test_tasks_json_legacy_seven_types():
    tasks = _load(TASKS_JSON)
    assert len(tasks) == 350
    counts = Counter(t["task_type"] for t in tasks)
    for tt in LEGACY_TASK_TYPE_CYCLE:
        assert counts[tt] == 50
    assert set(counts) == set(LEGACY_TASK_TYPE_CYCLE)
    for i, task in enumerate(tasks):
        assert task["task_type"] == LEGACY_TASK_TYPE_CYCLE[i % len(LEGACY_TASK_TYPE_CYCLE)]


def test_complete_json_ten_types():
    tasks = _load(COMPLETE_JSON)
    assert len(tasks) == 500
    counts = Counter(t["task_type"] for t in tasks)
    for tt in TASK_TYPE_CYCLE:
        assert counts[tt] == 50
    for i, task in enumerate(tasks):
        assert task["task_type"] == TASK_TYPE_CYCLE[i % len(TASK_TYPE_CYCLE)]
    for tt in CHALLENGE_TYPE_CYCLE:
        assert counts[tt] == 50


def test_complete_interleave_idempotent():
    tasks = _load(COMPLETE_JSON)
    assert interleave_tasks_by_type(tasks) == tasks


def test_max_tasks_includes_constancy_row():
    tasks = _load(COMPLETE_JSON)
    first_10 = tasks[:10]
    types = {t["task_type"] for t in first_10}
    assert "constancy_row" in types
    assert "tuple_grid" in types


def test_ravens_numerical_max_tasks_includes_constancy_row():
    task = RavensNumericalTask(
        max_tasks=10,
        prompt_type="instruction",
    )
    types = {s.metadata["task"]["task_type"] for s in task.canonical_stimuli()}
    assert "constancy_row" in types
    assert "distribution_of_three" in types
