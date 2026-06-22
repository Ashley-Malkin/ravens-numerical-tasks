#!/usr/bin/env python3
"""Convert all 2x2 matrices in tasks.json to 3x3 while preserving the same pattern."""

import json
import random
import sys
from pathlib import Path

from generator import generate_distractors


def is_2x2(matrix: list) -> bool:
    """Return True if matrix is 2 rows and 2 columns."""
    if not matrix or len(matrix) != 2:
        return False
    return all(len(row) == 2 for row in matrix)


def expand_constancy(matrix: list) -> list:
    """Constancy: all cells same value N, blank at (2,2)."""
    n = matrix[0][0]
    return [
        [n, n, n],
        [n, n, n],
        [n, n, None],
    ]


def expand_pattern(matrix: list) -> list:
    """Pattern: col0 constant x, col1 constant y; extend col2 as y. Blank (2,2)."""
    x, y = matrix[0][0], matrix[0][1]
    return [
        [x, y, y],
        [x, y, y],
        [x, y, None],
    ]


def expand_progression(matrix: list) -> list:
    """Progression: each row [k, k+1, k+2]. Blank at (2,2), answer is b+2."""
    a, b = matrix[0][0], matrix[1][0]
    return [
        [a, a + 1, a + 2],
        [b, b + 1, b + 2],
        [b, b + 1, None],
    ]


def expand_pattern_tuple(matrix: list) -> list:
    """Pattern tuple: diagonal (0,0)=a,(1,1)=a,(2,2)=?; (0,1)=(1,0)=(0,2)=(1,2)=(2,0)=(2,1)=b."""
    a, b = matrix[0][0], matrix[0][1]
    return [
        [a, b, b],
        [b, a, b],
        [b, b, None],
    ]


def expand_task(task: dict) -> dict:
    """If task has 2x2 matrix, expand to 3x3; else return unchanged."""
    matrix = task["matrix"]
    if not is_2x2(matrix):
        return task

    task_type = task.get("task_type", "")
    if task_type == "constancy":
        task = {**task, "matrix": expand_constancy(matrix)}
    elif task_type == "pattern":
        task = {**task, "matrix": expand_pattern(matrix)}
    elif task_type == "progression":
        new_matrix = expand_progression(matrix)
        b = new_matrix[2][0]
        correct = b + 2
        rng = random.Random()
        distractors = generate_distractors(correct, 1, 20, rng=rng)
        options = [correct] + distractors
        rng.shuffle(options)
        task = {
            **task,
            "matrix": new_matrix,
            "answer_options": options,
            "correct_index": options.index(correct),
        }
    elif task_type == "pattern_tuple":
        task = {**task, "matrix": expand_pattern_tuple(matrix)}
    # combine and intersection are already 3x3
    return task


def main() -> None:
    path = Path(__file__).parent / "tasks.json"
    with open(path) as f:
        data = json.load(f)

    tasks = data.get("tasks", data)
    expanded = [expand_task(t) for t in tasks]
    data["tasks"] = expanded

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    n_2x2 = sum(1 for t in tasks if is_2x2(t.get("matrix", [])))
    print(f"Expanded {n_2x2} 2x2 matrices to 3x3 in {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
