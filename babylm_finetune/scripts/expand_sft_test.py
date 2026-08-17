#!/usr/bin/env python3
"""DEPRECATED: prefer ``generate_sft_splits.py``, which builds magnitude-balanced
train/val/test together.

Legacy helper: expand SFT test.json to N examples per subtask without matrix
leakage. Keeps existing test items, then generates additional unique matrices
that are disjoint from train, val, the current test, and ``IN_CONTEXT_EXAMPLES``.

Writes:
  - babylm_finetune/data/test.json
  - data/tasks.json (kept in sync so Modal ravens eval uses the same set)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_sft_splits import (  # noqa: E402
    DEFAULT_MAX_VAL,
    DEFAULT_MIN_VAL,
    _generate_unique_for_type,
    _normalize_task,
    write_json_tasks,
)
from ravens_numerical.generation.generator import (  # noqa: E402
    TASK_TYPE_CYCLE,
    interleave_tasks_by_type,
    matrix_fingerprint,
    task_fingerprint,
)
from ravens_numerical.prompts.prompts import IN_CONTEXT_EXAMPLES  # noqa: E402

DEFAULT_SEED = 20260730
DEFAULT_PER_TYPE = 50


def _load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [_normalize_task(t) for t in data["tasks"]]


def _icl_matrices() -> set[str]:
    fps: set[str] = set()
    for examples in IN_CONTEXT_EXAMPLES.values():
        for ex in examples:
            fps.add(matrix_fingerprint({"matrix": ex["matrix"]}))
    return fps


def expand_test(
    *,
    data_dir: Path,
    repo_tasks_json: Path,
    per_type: int,
    seed: int,
    min_val: int,
    max_val: int,
    sync_repo_tasks: bool,
) -> list[dict]:
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    test_path = data_dir / "test.json"
    for p in (train_path, val_path, test_path):
        if not p.is_file():
            raise SystemExit(f"Missing required split file: {p}")

    train = _load_tasks(train_path)
    val = _load_tasks(val_path)
    test = _load_tasks(test_path)

    seen_matrices = {matrix_fingerprint(t) for t in train + val + test}
    seen_matrices |= _icl_matrices()
    seen_tasks = {task_fingerprint(t) for t in train + val + test}

    counts = Counter(t["task_type"] for t in test)
    for tt in TASK_TYPE_CYCLE:
        if counts[tt] > per_type:
            raise SystemExit(
                f"test already has {counts[tt]} {tt} tasks (> target {per_type})"
            )

    rng = random.Random(seed)
    expanded: list[dict] = list(test)
    added_by_type: dict[str, int] = {}

    for task_type in TASK_TYPE_CYCLE:
        need = per_type - counts[task_type]
        added_by_type[task_type] = need
        if need <= 0:
            continue
        extra = _generate_unique_for_type(
            task_type,
            need,
            rng,
            seen_matrices,
            seen_tasks,
            min_val=min_val,
            max_val=max_val,
        )
        expanded.extend(extra)

    expanded = interleave_tasks_by_type(expanded)

    # Final uniqueness / disjointness checks before write.
    fps = [matrix_fingerprint(t) for t in expanded]
    if len(fps) != len(set(fps)):
        raise SystemExit("Expanded test has duplicate matrices within itself")
    train_fps = {matrix_fingerprint(t) for t in train}
    val_fps = {matrix_fingerprint(t) for t in val}
    expanded_fps = set(fps)
    if expanded_fps & train_fps:
        raise SystemExit(
            f"Expanded test overlaps train "
            f"({len(expanded_fps & train_fps)} matrices)"
        )
    if expanded_fps & val_fps:
        raise SystemExit(
            f"Expanded test overlaps val ({len(expanded_fps & val_fps)} matrices)"
        )
    if expanded_fps & _icl_matrices():
        raise SystemExit("Expanded test overlaps IN_CONTEXT_EXAMPLES")

    final_counts = Counter(t["task_type"] for t in expanded)
    for tt in TASK_TYPE_CYCLE:
        if final_counts[tt] != per_type:
            raise SystemExit(
                f"Expected {per_type} {tt} tasks, got {final_counts[tt]}"
            )
    if len(expanded) != per_type * len(TASK_TYPE_CYCLE):
        raise SystemExit(
            f"Expected {per_type * len(TASK_TYPE_CYCLE)} test tasks, "
            f"got {len(expanded)}"
        )

    write_json_tasks(test_path, expanded)
    if sync_repo_tasks:
        write_json_tasks(repo_tasks_json, expanded)

    print(f"Wrote expanded test ({len(expanded)}) -> {test_path}")
    if sync_repo_tasks:
        print(f"Synced repo tasks.json -> {repo_tasks_json}")
    print(f"Per type: {dict(final_counts)}")
    print(f"Added per type: {added_by_type}")
    print(f"Digit range for new items: [{min_val}, {max_val}]")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand babylm_finetune/data/test.json to N unique examples per "
            "subtask, disjoint from train/val/ICL."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "data",
        help="Directory with train.json, val.json, test.json",
    )
    parser.add_argument(
        "--repo-tasks-json",
        type=Path,
        default=REPO_ROOT / "data" / "tasks.json",
        help="Also sync this file (Modal eval default)",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=DEFAULT_PER_TYPE,
        help=f"Target examples per subtask (default: {DEFAULT_PER_TYPE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for newly generated items (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_VAL,
        dest="min_val",
        help=f"Min digit for newly generated items (default: {DEFAULT_MIN_VAL})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_VAL,
        dest="max_val",
        help=f"Max digit for newly generated items (default: {DEFAULT_MAX_VAL})",
    )
    parser.add_argument(
        "--no-sync-repo-tasks",
        action="store_true",
        help="Do not overwrite data/tasks.json",
    )
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip check_overlaps.py after writing",
    )
    args = parser.parse_args()

    if args.per_type <= 0:
        raise SystemExit("--per-type must be positive")
    if args.min_val > args.max_val:
        raise SystemExit("--min must be <= --max")

    # Backup existing test before overwrite.
    data_dir = args.data_dir.resolve()
    test_path = data_dir / "test.json"
    backup = data_dir / "test_140_backup.json"
    if test_path.is_file() and not backup.is_file():
        shutil.copy2(test_path, backup)
        print(f"Backed up previous test -> {backup}")

    expand_test(
        data_dir=data_dir,
        repo_tasks_json=args.repo_tasks_json.resolve(),
        per_type=args.per_type,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        sync_repo_tasks=not args.no_sync_repo_tasks,
    )

    if not args.skip_overlap_check:
        from check_overlaps import check_overlaps

        check_overlaps(data_dir)


if __name__ == "__main__":
    main()
