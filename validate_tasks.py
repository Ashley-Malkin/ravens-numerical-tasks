#!/usr/bin/env python3
"""Validate tasks.json: structure, semantic correctness, correct_index, correct_letter."""

import json
import sys
from pathlib import Path

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

        else:
            errs.append(f"task {i}: unknown task_type {tt!r}")

    except Exception as e:
        errs.append(f"task {i} ({tt}): exception {e!r}")

    return errs


def main() -> int:
    path = Path(__file__).resolve().parent / "tasks.json"
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data)
    all_errs: list[str] = []
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
