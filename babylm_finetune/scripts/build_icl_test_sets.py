#!/usr/bin/env python3
"""Build oneICL_*/threeICL_* JSON with *per-task* ICL demos.

Each task gets its own ICL list (1 or 3 demos of the same type). Demos are
matrix-disjoint from train / val / test, from ``IN_CONTEXT_EXAMPLES``, and from
every other ICL demo (including other tasks of the same type). Within a task,
``oneICL`` demos are the first demo of that task's ``threeICL`` list.

Also writes SFT completion JSONL under ``babylm_finetune/data/sft/`` for train
and val, and syncs task JSON copies under ``data/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ravens_numerical.generation.generator import (  # noqa: E402
    TASK_TYPE_CYCLE,
    generate_in_context_examples,
    matrix_fingerprint,
    task_fingerprint,
)
from ravens_numerical.prompts.prompts import (  # noqa: E402
    IN_CONTEXT_EXAMPLES,
    build_completion_prompt,
    expected_completion_answer,
)

DEFAULT_SEED = 20260804
# Wide ranges so we can mint hundreds of unique demos per type.
ICL_MIN_VAL = 1
ICL_MAX_VAL = 5000
ICL_DIGIT_MIN = 0
ICL_DIGIT_MAX = 49
SPLITS = ("train", "val", "test")
SFT_SPLITS = ("train", "val")
N_THREE = 3


def _load_payload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_tasks(path: Path) -> list[dict]:
    data = _load_payload(path)
    return list(data["tasks"])


def _correct_index(ex: dict) -> int:
    if "correct_index" in ex:
        return int(ex["correct_index"])
    return ord(str(ex["correct_letter"]).upper()[0]) - ord("A")


def _exclude_fingerprints(*task_lists: list[dict]) -> tuple[set[str], set[str]]:
    fps: set[str] = set()
    mats: set[str] = set()
    for tasks in task_lists:
        for t in tasks:
            fps.add(task_fingerprint(t))
            mats.add(matrix_fingerprint(t))
    for examples in IN_CONTEXT_EXAMPLES.values():
        for ex in examples:
            mats.add(matrix_fingerprint({"matrix": ex["matrix"]}))
            fps.add(
                task_fingerprint(
                    {
                        "matrix": ex["matrix"],
                        "answer_options": ex["answer_options"],
                        "correct_index": _correct_index(ex),
                    }
                )
            )
    return fps, mats


def _counts_needed_per_type(
    split_tasks: dict[str, list[dict]], n_per_task: int
) -> dict[str, int]:
    """Total demos needed per type across all splits for ``n_per_task`` ICL."""
    needed: dict[str, int] = {tt: 0 for tt in TASK_TYPE_CYCLE}
    for tasks in split_tasks.values():
        for t in tasks:
            needed[t["task_type"]] += n_per_task
    return needed


def _assign_per_task_icl(
    tasks: list[dict],
    *,
    pools: dict[str, list[dict]],
    cursors: dict[str, int],
    n_per_task: int,
) -> list[dict]:
    """Copy tasks and attach ``icl`` demos drawn uniquely from ``pools``."""
    out: list[dict] = []
    for task in tasks:
        tt = task["task_type"]
        start = cursors[tt]
        end = start + n_per_task
        demos = pools[tt][start:end]
        if len(demos) != n_per_task:
            raise RuntimeError(
                f"ICL pool exhausted for {tt}: need {n_per_task} at cursor {start}"
            )
        cursors[tt] = end
        item = dict(task)
        item["icl"] = demos
        out.append(item)
    return out


def _write_icl_split(
    path: Path,
    *,
    tasks: list[dict],
    n_icl_per_type: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks": tasks,
        "n_icl_per_type": n_icl_per_type,
        "icl_mode": "per_task",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(tasks)} tasks, {n_icl_per_type} ICL/task)")


def _task_to_icl_sft_row(task: dict, *, n_examples: int) -> dict:
    """Completion-only SFT row using this task's own ``icl`` demos."""
    demos = list(task.get("icl") or [])
    icl_bank = {task["task_type"]: demos}
    prompt = build_completion_prompt(
        task, n_examples=n_examples, icl_examples=icl_bank
    )
    answer = expected_completion_answer(task)
    return {
        "prompt": prompt,
        "completion": answer + "]",
        "task_type": task["task_type"],
        "correct_index": task["correct_index"],
        "correct_letter": task["correct_letter"],
        "answer_options": task["answer_options"],
        "matrix": task["matrix"],
    }


def _write_icl_sft_jsonl(
    path: Path,
    *,
    tasks: list[dict],
    n_icl_per_type: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            row = _task_to_icl_sft_row(task, n_examples=n_icl_per_type)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(tasks)} rows, {n_icl_per_type} ICL/task)")


def build_icl_split_sets(
    data_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    sync_data_dir: Path | None = None,
    sft_dir: Path | None = None,
    icl_min_val: int = ICL_MIN_VAL,
    icl_max_val: int = ICL_MAX_VAL,
    icl_digit_min: int = ICL_DIGIT_MIN,
    icl_digit_max: int = ICL_DIGIT_MAX,
) -> None:
    split_tasks = {
        name: _load_tasks(data_dir / f"{name}.json") for name in SPLITS
    }
    exclude_fps, exclude_mats = _exclude_fingerprints(
        split_tasks["train"], split_tasks["val"], split_tasks["test"]
    )

    needed = _counts_needed_per_type(split_tasks, N_THREE)
    n_pool = max(needed.values())
    print(
        f"Generating {n_pool} unique ICL demos/type "
        f"(mag=[{icl_min_val},{icl_max_val}], "
        f"digits=[{icl_digit_min},{icl_digit_max}])..."
    )
    pools = generate_in_context_examples(
        exclude_fps,
        exclude_matrices=exclude_mats,
        n_per_type=n_pool,
        seed=seed,
        min_val=icl_min_val,
        max_val=icl_max_val,
        digit_min=icl_digit_min,
        digit_max=icl_digit_max,
    )
    # Trim each type to exactly what we need (generator makes a uniform n).
    for tt in TASK_TYPE_CYCLE:
        pools[tt] = pools[tt][: needed[tt]]
        print(f"  {tt}: {len(pools[tt])} demos")

    cursors = {tt: 0 for tt in TASK_TYPE_CYCLE}
    # Assign in split order so train/val/test pools are contiguous & disjoint.
    tasks_three: dict[str, list[dict]] = {}
    for split in SPLITS:
        tasks_three[split] = _assign_per_task_icl(
            split_tasks[split],
            pools=pools,
            cursors=cursors,
            n_per_task=N_THREE,
        )
    assert all(cursors[tt] == needed[tt] for tt in TASK_TYPE_CYCLE)

    tasks_one: dict[str, list[dict]] = {}
    for split in SPLITS:
        one_tasks: list[dict] = []
        for task in tasks_three[split]:
            item = dict(task)
            item["icl"] = task["icl"][:1]
            one_tasks.append(item)
        tasks_one[split] = one_tasks

    out_dirs = [data_dir]
    if sync_data_dir is not None:
        sync_data_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(sync_data_dir)

    for split in SPLITS:
        for out_dir in out_dirs:
            _write_icl_split(
                out_dir / f"oneICL_{split}.json",
                tasks=tasks_one[split],
                n_icl_per_type=1,
            )
            _write_icl_split(
                out_dir / f"threeICL_{split}.json",
                tasks=tasks_three[split],
                n_icl_per_type=3,
            )

    if sft_dir is not None:
        sft_dir.mkdir(parents=True, exist_ok=True)
        for split in SFT_SPLITS:
            _write_icl_sft_jsonl(
                sft_dir / f"oneICL_{split}.jsonl",
                tasks=tasks_one[split],
                n_icl_per_type=1,
            )
            _write_icl_sft_jsonl(
                sft_dir / f"threeICL_{split}.jsonl",
                tasks=tasks_three[split],
                n_icl_per_type=3,
            )


# Back-compat alias.
build_icl_test_sets = build_icl_split_sets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build oneICL_/threeICL_{train,val,test}.json with unique per-task "
            "ICL demos, plus SFT JSONL for train/val"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "data",
        help="Directory with train.json / val.json / test.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for ICL generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--no-sync-data",
        action="store_true",
        help="Do not also write copies under data/",
    )
    parser.add_argument(
        "--no-sft-jsonl",
        action="store_true",
        help="Do not write babylm_finetune/data/sft/oneICL_*.jsonl etc.",
    )
    parser.add_argument(
        "--icl-min",
        type=int,
        default=ICL_MIN_VAL,
        help=f"Min magnitude for ICL demos (default: {ICL_MIN_VAL})",
    )
    parser.add_argument(
        "--icl-max",
        type=int,
        default=ICL_MAX_VAL,
        help=f"Max magnitude for ICL demos (default: {ICL_MAX_VAL})",
    )
    parser.add_argument(
        "--icl-digit-min",
        type=int,
        default=ICL_DIGIT_MIN,
        help=f"Min digit for tuple/logic ICL (default: {ICL_DIGIT_MIN})",
    )
    parser.add_argument(
        "--icl-digit-max",
        type=int,
        default=ICL_DIGIT_MAX,
        help=f"Max digit for tuple/logic ICL (default: {ICL_DIGIT_MAX})",
    )
    args = parser.parse_args()
    sync = None if args.no_sync_data else REPO_ROOT / "data"
    sft = None if args.no_sft_jsonl else (
        REPO_ROOT / "babylm_finetune" / "data" / "sft"
    )
    build_icl_split_sets(
        args.data_dir.resolve(),
        seed=args.seed,
        sync_data_dir=sync,
        sft_dir=sft,
        icl_min_val=args.icl_min,
        icl_max_val=args.icl_max,
        icl_digit_min=args.icl_digit_min,
        icl_digit_max=args.icl_digit_max,
    )


if __name__ == "__main__":
    main()
