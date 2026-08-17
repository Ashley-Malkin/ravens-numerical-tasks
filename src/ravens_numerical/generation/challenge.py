"""Build ``data/challenge_tasks.json``: harder Raven-style numerical matrices.

Three task types × 50 items (150 total), round-robin interleaved:

- ``distribution_of_three`` — cyclic ordering of three values (Latin-row shifts)
- ``progression_plus_n`` — per-row arithmetic progression with step in
  ``{2, 3, 5, 7, 10}``
- ``tuple_grid`` — cells are ``(row_key, col_key)`` pairs; order-variant answers
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ravens_numerical.generation.generator import (
    CHALLENGE_TYPE_CYCLE,
    PROGRESSION_STEPS,
    generate_distribution_of_three_task,
    generate_progression_plus_n_task,
    generate_tuple_grid_task,
    interleave_tasks_by_type,
    matrix_fingerprint,
    task_fingerprint,
)
from ravens_numerical.paths import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "data" / "challenge_tasks.json"
LETTERS = "ABCD"
N_PER_TYPE = 50


def _with_letter(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    out["correct_letter"] = LETTERS[int(out["correct_index"])]
    return out


def _accept_unique(
    task: dict[str, Any],
    *,
    seen_fp: set[str],
    seen_mx: set[str],
) -> bool:
    fp = task_fingerprint(task)
    mf = matrix_fingerprint(task)
    if fp in seen_fp or mf in seen_mx:
        return False
    seen_fp.add(fp)
    seen_mx.add(mf)
    return True


def build_challenge_tasks(
    *,
    n_per_type: int = N_PER_TYPE,
    seed: int = 20260812,
    max_attempts_per_item: int = 500,
) -> list[dict[str, Any]]:
    """Generate unique challenge items and interleave by type."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {
        tt: [] for tt in CHALLENGE_TYPE_CYCLE
    }
    seen_fp: set[str] = set()
    seen_mx: set[str] = set()

    for _ in range(n_per_type):
        for attempt in range(max_attempts_per_item):
            task = _with_letter(generate_distribution_of_three_task(rng=rng))
            if _accept_unique(task, seen_fp=seen_fp, seen_mx=seen_mx):
                by_type["distribution_of_three"].append(task)
                break
        else:
            raise RuntimeError("Could not sample unique distribution_of_three")

    steps = list(PROGRESSION_STEPS) * (n_per_type // len(PROGRESSION_STEPS))
    rem = n_per_type % len(PROGRESSION_STEPS)
    steps.extend(list(PROGRESSION_STEPS)[:rem])
    rng.shuffle(steps)
    for step in steps:
        for attempt in range(max_attempts_per_item):
            task = _with_letter(
                generate_progression_plus_n_task(step=step, rng=rng)
            )
            if _accept_unique(task, seen_fp=seen_fp, seen_mx=seen_mx):
                by_type["progression_plus_n"].append(task)
                break
        else:
            raise RuntimeError(
                f"Could not sample unique progression_plus_n for step={step}"
            )

    for _ in range(n_per_type):
        for attempt in range(max_attempts_per_item):
            task = _with_letter(generate_tuple_grid_task(rng=rng))
            if _accept_unique(task, seen_fp=seen_fp, seen_mx=seen_mx):
                by_type["tuple_grid"].append(task)
                break
        else:
            raise RuntimeError("Could not sample unique tuple_grid")

    flat = [t for tt in CHALLENGE_TYPE_CYCLE for t in by_type[tt]]
    return interleave_tasks_by_type(flat, type_cycle=CHALLENGE_TYPE_CYCLE)


def write_challenge_tasks(
    output: Path | None = None,
    *,
    n_per_type: int = N_PER_TYPE,
    seed: int = 20260812,
) -> Path:
    """Write ``challenge_tasks.json``; return the output path."""
    out = output or DEFAULT_OUT
    tasks = build_challenge_tasks(n_per_type=n_per_type, seed=seed)
    payload = {
        "tasks": tasks,
        "meta": {
            "purpose": "Harder Raven-style challenge suite",
            "n_per_type": n_per_type,
            "task_types": list(CHALLENGE_TYPE_CYCLE),
            "progression_steps": list(PROGRESSION_STEPS),
            "seed": seed,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data/challenge_tasks.json (3 types × 50 items)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--n-per-type",
        type=int,
        default=N_PER_TYPE,
        help=f"Items per task type (default: {N_PER_TYPE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260812,
        help="RNG seed (default: 20260812)",
    )
    args = parser.parse_args()
    path = write_challenge_tasks(
        args.output, n_per_type=args.n_per_type, seed=args.seed
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    from collections import Counter

    counts = Counter(t["task_type"] for t in data["tasks"])
    print(f"Wrote {len(data['tasks'])} tasks → {path}")
    print("Counts:", dict(counts))
    if "progression_plus_n" in counts:
        step_counts = Counter(
            t["step"] for t in data["tasks"] if t["task_type"] == "progression_plus_n"
        )
        print("progression_plus_n steps:", dict(sorted(step_counts.items())))


if __name__ == "__main__":
    main()
