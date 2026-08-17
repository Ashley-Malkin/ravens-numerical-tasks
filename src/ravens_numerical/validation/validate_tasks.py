#!/usr/bin/env python3
"""Validate complete.json / tasks.json: structure, semantic correctness, letters."""

from __future__ import annotations

import json
import sys

from ravens_numerical.generation.generator import LEGACY_TASK_TYPE_CYCLE, TASK_TYPE_CYCLE
from ravens_numerical.paths import COMPLETE_JSON, TASKS_JSON

LETTERS = "ABCD"


def norm_cell(c):
    if c is None:
        return None
    if isinstance(c, list) and len(c) == 1:
        return c[0]
    return c


def validate_task(i: int, task: dict) -> list[str]:
    errs: list[str] = []
    tt = task.get("task_type", "?")
    matrix = task.get("matrix")
    opts = task.get("answer_options")
    ci = task.get("correct_index")

    if matrix is None:
        return [f"task {i}: missing matrix"]
    if not isinstance(opts, list) or len(opts) != 4:
        errs.append(f"task {i} ({tt}): answer_options must have length 4")
        return errs
    if ci is None or not (0 <= ci < 4):
        errs.append(f"task {i} ({tt}): correct_index must be 0..3")
        return errs

    picked = opts[ci]
    cl = task.get("correct_letter")
    if cl is None:
        errs.append(f"task {i} ({tt}): missing correct_letter")
    elif cl != LETTERS[ci]:
        errs.append(
            f"task {i} ({tt}): correct_letter {cl!r} != {LETTERS[ci]!r} for correct_index={ci}"
        )

    try:
        if tt == "constancy":
            n = None
            for r, row in enumerate(matrix):
                for c, cell in enumerate(row):
                    if cell is None:
                        if (r, c) != (2, 2):
                            errs.append(f"task {i}: constancy null only at (2,2)")
                    else:
                        v = norm_cell(cell)
                        if n is None:
                            n = v
                        elif v != n:
                            errs.append(f"task {i}: constancy mismatch {v} vs {n}")
            if n is not None and picked != n:
                errs.append(f"task {i}: constancy picked {picked} expected {n}")

        elif tt == "constancy_row":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: constancy_row expects 3x3")
            else:
                row_vals: list[int | None] = []
                for r, row in enumerate(matrix):
                    vals = [norm_cell(c) for c in row if c is not None]
                    if not vals:
                        errs.append(f"task {i}: constancy_row row {r} has no filled cells")
                        continue
                    if len(set(vals)) != 1:
                        errs.append(f"task {i}: constancy_row row {r} not constant")
                    else:
                        row_vals.append(vals[0])
                if matrix[2][2] is not None:
                    errs.append(f"task {i}: constancy_row blank must be null at (2,2)")
                if len(row_vals) == 3 and picked != row_vals[2]:
                    errs.append(
                        f"task {i}: constancy_row picked {picked} expected {row_vals[2]}"
                    )

        elif tt == "pattern":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: pattern expects 3x3")
            else:
                x, y = matrix[0][0], matrix[0][1]
                exp = [
                    [x, y, y],
                    [x, y, y],
                    [x, y, None],
                ]
                if matrix != exp:
                    errs.append(f"task {i}: pattern matrix shape mismatch")
                if picked != y:
                    errs.append(f"task {i}: pattern picked {picked} expected {y}")

        elif tt == "pattern_tuple":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: pattern_tuple expects 3x3")
            else:
                a, b = matrix[0][0], matrix[0][1]
                exp = [[a, b, b], [b, a, b], [b, b, None]]
                if matrix != exp:
                    errs.append(f"task {i}: pattern_tuple matrix mismatch")
                if picked != a:
                    errs.append(f"task {i}: pattern_tuple picked {picked} expected {a}")

        elif tt == "progression":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: progression expects 3x3")
            else:
                a, b = matrix[0][0], matrix[1][0]
                exp0 = [a, a + 1, a + 2]
                exp1 = [b, b + 1, b + 2]
                exp2 = [b, b + 1, None]
                if matrix[0] != exp0 or matrix[1] != exp1 or matrix[2] != exp2:
                    errs.append(f"task {i}: progression row pattern mismatch")
                want = b + 2
                if picked != want:
                    errs.append(f"task {i}: progression picked {picked} expected {want}")

        elif tt == "combine":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: combine expects 3x3")
            else:
                for r in range(3):
                    if r < 2:
                        c0, c1, c2 = matrix[r][0], matrix[r][1], matrix[r][2]
                        if c2 != list(c0) + list(c1):
                            errs.append(f"task {i}: combine row {r} col3 != concat col1 col2")
                    else:
                        a3, b3 = matrix[2][0], matrix[2][1]
                        want = list(a3) + list(b3)
                        if picked != want:
                            errs.append(f"task {i}: combine picked {picked} expected {want}")

        elif tt == "intersection":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: intersection expects 3x3")
            else:
                for r in range(2):
                    L, R, ccell = matrix[r][0], matrix[r][1], matrix[r][2]
                    if not isinstance(L, list) or len(L) != 2:
                        errs.append(f"task {i}: intersection row{r} left not pair")
                    if not isinstance(R, list) or len(R) != 2:
                        errs.append(f"task {i}: intersection row{r} right not pair")
                    if not isinstance(ccell, list) or len(ccell) != 1:
                        errs.append(f"task {i}: intersection row{r} col3 not [c]")
                    inter = set(L) & set(R)
                    if inter != {ccell[0]}:
                        errs.append(
                            f"task {i}: intersection row{r} overlap set {inter} vs [c]={ccell}"
                        )
                L2, R2 = matrix[2][0], matrix[2][1]
                if matrix[2][2] is not None:
                    errs.append(f"task {i}: intersection blank must be null")
                inter2 = set(L2) & set(R2)
                if len(inter2) != 1:
                    errs.append(f"task {i}: intersection last row overlap set {inter2}")
                else:
                    want = next(iter(inter2))
                    if picked != want:
                        errs.append(f"task {i}: intersection picked {picked} expected {want}")

        elif tt == "distribution_of_three":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: distribution_of_three expects 3x3")
            else:
                a, b, c = matrix[0]
                exp = [[a, b, c], [b, c, a], [c, a, None]]
                if matrix != exp:
                    errs.append(f"task {i}: distribution_of_three row shift mismatch")
                if picked != b:
                    errs.append(
                        f"task {i}: distribution_of_three picked {picked} expected {b}"
                    )
                others = [o for o in opts if o != picked]
                from_set = [o for o in others if o in {a, b, c}]
                if len(from_set) != 2:
                    errs.append(
                        f"task {i}: distribution_of_three expected 2 distractors "
                        f"from the triple, got {from_set}"
                    )

        elif tt == "progression_plus_n":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: progression_plus_n expects 3x3")
            else:
                step = matrix[0][1] - matrix[0][0]
                if step <= 0:
                    errs.append(f"task {i}: progression_plus_n non-positive step {step}")
                for r in range(2):
                    if matrix[r][1] - matrix[r][0] != step or matrix[r][2] - matrix[r][1] != step:
                        errs.append(f"task {i}: progression_plus_n row {r} step mismatch")
                if matrix[2][1] - matrix[2][0] != step:
                    errs.append(f"task {i}: progression_plus_n last-row step mismatch")
                want = matrix[2][0] + 2 * step
                if picked != want:
                    errs.append(
                        f"task {i}: progression_plus_n picked {picked} expected {want}"
                    )

        elif tt == "tuple_grid":
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                errs.append(f"task {i}: tuple_grid expects 3x3")
            else:
                row_keys = [matrix[r][0][0] for r in range(3)]
                col_keys = [matrix[0][c][1] for c in range(3)]
                for r in range(3):
                    for c in range(3):
                        cell = matrix[r][c]
                        if r == 2 and c == 2:
                            if cell is not None:
                                errs.append(f"task {i}: tuple_grid blank must be null")
                            continue
                        if not isinstance(cell, list) or len(cell) != 2:
                            errs.append(f"task {i}: tuple_grid cell ({r},{c}) not a pair")
                        elif cell[0] != row_keys[r] or cell[1] != col_keys[c]:
                            errs.append(
                                f"task {i}: tuple_grid cell ({r},{c})={cell} "
                                f"expected {[row_keys[r], col_keys[c]]}"
                            )
                want = [row_keys[2], col_keys[2]]
                if picked != want:
                    errs.append(f"task {i}: tuple_grid picked {picked} expected {want}")

        else:
            errs.append(f"task {i}: unknown task_type {tt!r}")

    except Exception as e:
        errs.append(f"task {i} ({tt}): exception {e!r}")

    return errs


def validate_interleaved_order(
    tasks: list[dict],
    *,
    type_cycle: tuple[str, ...] | None = None,
) -> list[str]:
    """Ensure round-robin order so max_tasks slices cover all types."""
    cycle = type_cycle or TASK_TYPE_CYCLE
    errs: list[str] = []
    for i, task in enumerate(tasks):
        expected = cycle[i % len(cycle)]
        actual = task.get("task_type")
        if actual != expected:
            errs.append(
                f"task {i}: expected task_type {expected!r} in round-robin order, got {actual!r}"
            )
            break
    return errs


def main() -> int:
    path = COMPLETE_JSON if COMPLETE_JSON.is_file() else TASKS_JSON
    cycle = TASK_TYPE_CYCLE if path == COMPLETE_JSON else LEGACY_TASK_TYPE_CYCLE
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data)
    all_errs: list[str] = []
    all_errs.extend(validate_interleaved_order(tasks, type_cycle=cycle))
    for i, task in enumerate(tasks):
        all_errs.extend(validate_task(i, task))
    if all_errs:
        for e in all_errs:
            print(e, file=sys.stderr)
        print(f"FAILED: {len(all_errs)} issue(s)", file=sys.stderr)
        return 1
    print(f"OK: {len(tasks)} tasks validated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
