#!/usr/bin/env python3
"""Ensure ICL demos in ravens_prompts do not overlap tasks.json test items."""

from __future__ import annotations

import json
from pathlib import Path

from generator import matrix_fingerprint, task_fingerprint
from ravens_prompts import IN_CONTEXT_EXAMPLES
from validate_tasks import validate_task

REPO = Path(__file__).resolve().parent


def test_icl_examples_disjoint_from_tasks_json() -> None:
    tasks = json.loads((REPO / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    task_fps = {task_fingerprint(t) for t in tasks}
    matrix_fps = {matrix_fingerprint(t) for t in tasks}

    for task_type, examples in IN_CONTEXT_EXAMPLES.items():
        assert len(examples) == 3, f"{task_type}: expected 3 ICL examples"
        for i, ex in enumerate(examples):
            pseudo = {
                "matrix": ex["matrix"],
                "answer_options": ex["answer_options"],
                "correct_index": ord(ex["correct_letter"]) - ord("A"),
            }
            fp = task_fingerprint(pseudo)
            mf = matrix_fingerprint(pseudo)
            assert fp not in task_fps, f"{task_type}[{i}] duplicates a tasks.json item"
            assert mf not in matrix_fps, f"{task_type}[{i}] matrix duplicates a tasks.json item"


def test_icl_examples_validate() -> None:
    """Run validate_tasks semantics on each ICL demo."""
    for task_type, examples in IN_CONTEXT_EXAMPLES.items():
        for i, ex in enumerate(examples):
            task = {
                "task_type": task_type,
                "matrix": ex["matrix"],
                "answer_options": ex["answer_options"],
                "correct_index": ord(ex["correct_letter"]) - ord("A"),
                "correct_letter": ex["correct_letter"],
            }
            errs = validate_task(i, task)
            assert errs == [], f"{task_type}[{i}]: {errs}"


if __name__ == "__main__":
    test_icl_examples_disjoint_from_tasks_json()
    test_icl_examples_validate()
    print("OK: ICL examples are valid and disjoint from tasks.json")
