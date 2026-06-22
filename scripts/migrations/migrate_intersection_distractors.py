#!/usr/bin/env python3
"""Rebuild intersection answer_options using generate_distractors (matches generator)."""

import json
import random
import sys
from pathlib import Path

from generator import generate_distractors, generate_intersection_task

LETTERS = "ABCD"


def _matrix_digit_bounds(matrix: list) -> tuple[int, int]:
    lo: int | None = None
    hi: int | None = None
    for row in matrix:
        for cell in row:
            if cell is None:
                continue
            if isinstance(cell, list):
                for v in cell:
                    lo = v if lo is None else min(lo, v)
                    hi = v if hi is None else max(hi, v)
            else:
                lo = cell if lo is None else min(lo, cell)
                hi = cell if hi is None else max(hi, cell)
    assert lo is not None and hi is not None
    return lo, hi


def main() -> None:
    path = Path(__file__).resolve().parent / "tasks.json"
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data)
    n_fix = 0
    n_regen = 0
    for i, t in enumerate(tasks):
        if t.get("task_type") != "intersection":
            continue
        m = t["matrix"]
        inter = set(m[2][0]) & set(m[2][1])
        if len(inter) != 1:
            rng = random.Random(91_000 + i)
            nt = generate_intersection_task(0, 9, rng=rng)
            nt["correct_letter"] = LETTERS[nt["correct_index"]]
            tasks[i] = nt
            n_regen += 1
            continue
        correct = next(iter(inter))
        lo, hi = _matrix_digit_bounds(m)
        rng = random.Random(91_000 + i)
        distractors = generate_distractors(correct, lo, hi, rng=rng)
        if len(distractors) < 3:
            rng2 = random.Random(91_000 + i)
            nt = generate_intersection_task(lo, hi, rng=rng2)
            nt["correct_letter"] = LETTERS[nt["correct_index"]]
            tasks[i] = nt
            n_regen += 1
            continue
        options = [correct] + distractors
        rng.shuffle(options)
        t["answer_options"] = options
        t["correct_index"] = options.index(correct)
        t["correct_letter"] = LETTERS[t["correct_index"]]
        n_fix += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(
        f"intersection: {n_fix} in-place option rebuilds, {n_regen} full regenerations -> {path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
