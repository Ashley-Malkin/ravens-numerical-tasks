#!/usr/bin/env python3
"""Fix progression tasks: correct answer for (2,2) is row[2][0] + 2 (pattern continues +1)."""

import json
import random
import sys
from pathlib import Path

from generator import generate_distractors


def is_progression_3x3_with_blank_at_22(task: dict) -> bool:
    """True if progression task with 3x3 matrix and blank at (2,2)."""
    if task.get("task_type") != "progression":
        return False
    matrix = task.get("matrix", [])
    if len(matrix) != 3 or not all(len(row) == 3 for row in matrix):
        return False
    row2 = matrix[2]
    return row2[0] is not None and row2[1] is not None and row2[2] is None


def fix_progression_task(task: dict, min_val: int = 1, max_val: int = 20, rng: random.Random = None) -> dict:
    """Set correct answer to row[2][0] + 2 and regenerate options."""
    rng = rng or random.Random()
    matrix = task["matrix"]
    b = matrix[2][0]
    correct = b + 2

    distractors = generate_distractors(correct, min_val, max_val, rng=rng)
    options = [correct] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct)

    return {
        **task,
        "answer_options": options,
        "correct_index": correct_index,
    }


def main() -> None:
    path = Path(__file__).parent / "tasks.json"
    with open(path) as f:
        data = json.load(f)

    tasks = data.get("tasks", data)
    rng = random.Random(42)
    fixed = []
    n_fixed = 0
    for t in tasks:
        if is_progression_3x3_with_blank_at_22(t):
            fixed.append(fix_progression_task(t, rng=rng))
            n_fixed += 1
        else:
            fixed.append(t)
    data["tasks"] = fixed

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Fixed {n_fixed} progression tasks (correct = row[2][0] + 2)", file=sys.stderr)


if __name__ == "__main__":
    main()
