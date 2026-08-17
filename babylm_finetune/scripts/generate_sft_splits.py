#!/usr/bin/env python3
"""Generate train/val/test Raven SFT splits with balanced small/large magnitudes.

Scalar task types (constancy, constancy_row, pattern, progression) are drawn
half from a small digit band and half from a large band, and those halves are
allocated evenly into train / val / test. Digit-tuple types (combine,
intersection, pattern_tuple) stay in 0–9 and are only uniqueness-stratified.

All three splits are matrix-disjoint from each other and from
``IN_CONTEXT_EXAMPLES``. Test is also written to ``data/complete.json``.
``data/tasks.json`` (7-type backup) is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ravens_numerical.generation.generator import (  # noqa: E402
    LETTERS,
    CHALLENGE_TYPE_CYCLE,
    LEGACY_TASK_TYPE_CYCLE,
    PROGRESSION_STEPS,
    TASK_TYPE_CYCLE,
    _expand_task_to_3x3,
    generate_combine_task,
    generate_constancy_row_task,
    generate_constancy_task,
    generate_distribution_of_three_task,
    generate_distractors,
    generate_intersection_task,
    generate_pattern_task,
    generate_pattern_tuple_task,
    generate_progression_plus_n_task,
    generate_tuple_grid_task,
    interleave_tasks_by_type,
    matrix_fingerprint,
    task_fingerprint,
)
from ravens_numerical.paths import CHALLENGE_TASKS_JSON, COMPLETE_JSON  # noqa: E402
from ravens_numerical.prompts.prompts import (  # noqa: E402
    IN_CONTEXT_EXAMPLES,
    expected_completion_answer,
    matrix_to_completion_query,
)

DEFAULT_SEED = 20260731
TRAIN_PER_TYPE = 180
VAL_PER_TYPE = 20
TEST_PER_TYPE = 50
DEFAULT_PILOT_SIZES = (10, 50, 100)

# Scalar types support a real magnitude axis in [1, 500].
MAGNITUDE_TYPES = frozenset(
    {
        "constancy",
        "constancy_row",
        "pattern",
        "progression",
        "distribution_of_three",
        "progression_plus_n",
        "tuple_grid",
    }
)
# Digit / tuple types are confined to 0–9 (no small/large band).
DIGIT_TYPES = frozenset({"combine", "intersection", "pattern_tuple"})
# Frozen eval types whose test items come from challenge_tasks.json.
CHALLENGE_TYPES = frozenset(CHALLENGE_TYPE_CYCLE)

DEFAULT_MIN_VAL = 1
DEFAULT_MAX_VAL = 500
# Inclusive bands; split the full range in half.
DEFAULT_SMALL_MAX = 250  # small = [min, small_max]
# large = [small_max + 1, max]


def _load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return list(data["tasks"])


def _normalize_task(task: dict) -> dict:
    """Ensure tasks.json schema fields (including correct_letter)."""
    out = {
        "task_type": task["task_type"],
        "matrix": task["matrix"],
        "answer_options": task["answer_options"],
        "correct_index": int(task["correct_index"]),
    }
    out["correct_letter"] = LETTERS[out["correct_index"]]
    for key in ("step", "perm_invariant"):
        if key in task:
            out[key] = task[key]
    return out


def task_answer_magnitude(task: dict) -> int | None:
    """Scalar answer magnitude, or ``None`` for tuple answers.

    ``tuple_grid`` answers are pairs; use the row-key (first coordinate).
    """
    ans = task["answer_options"][task["correct_index"]]
    if isinstance(ans, list):
        if task.get("task_type") == "tuple_grid" and ans:
            return int(ans[0])
        return None
    return int(ans)


def magnitude_band(
    task: dict, *, small_max: int
) -> str | None:
    """Return ``small`` / ``large`` for scalar magnitude types; else ``None``."""
    if task["task_type"] not in MAGNITUDE_TYPES:
        return None
    mag = task_answer_magnitude(task)
    if mag is None:
        return None
    return "small" if mag <= small_max else "large"


def generate_candidate(
    task_type: str,
    rng: random.Random,
    *,
    min_val: int,
    max_val: int,
    step: int | None = None,
) -> dict:
    """Generate one 3×3 task matching the ``tasks.json`` layout."""
    if task_type == "constancy":
        return _expand_task_to_3x3(
            generate_constancy_task(min_val=min_val, max_val=max_val, rng=rng)
        )
    if task_type == "constancy_row":
        return generate_constancy_row_task(min_val=min_val, max_val=max_val, rng=rng)
    if task_type == "pattern":
        return _expand_task_to_3x3(
            generate_pattern_task(min_val=min_val, max_val=max_val, rng=rng)
        )
    if task_type == "pattern_tuple":
        return _expand_task_to_3x3(generate_pattern_tuple_task(rng=rng))
    if task_type == "progression":
        if max_val - min_val < 2:
            raise ValueError("progression needs max_val >= min_val + 2")
        hi = max_val - 2
        b = rng.randint(min_val, hi)
        a = rng.randint(min_val, hi)
        correct = b + 2
        distractors = generate_distractors(correct, min_val, max_val, rng=rng)
        options = [correct] + distractors
        rng.shuffle(options)
        return {
            "task_type": "progression",
            "matrix": [[a, a + 1, a + 2], [b, b + 1, b + 2], [b, b + 1, None]],
            "answer_options": options,
            "correct_index": options.index(correct),
        }
    if task_type == "combine":
        return generate_combine_task(rng=rng)
    if task_type == "intersection":
        return generate_intersection_task(
            digit_min=min(min_val, 0), digit_max=min(max_val, 9), rng=rng
        )
    if task_type == "distribution_of_three":
        return generate_distribution_of_three_task(
            min_val=min_val, max_val=max_val, rng=rng
        )
    if task_type == "progression_plus_n":
        st = step if step is not None else rng.choice(PROGRESSION_STEPS)
        return generate_progression_plus_n_task(
            step=st, min_val=min_val, max_val=max_val, rng=rng
        )
    if task_type == "tuple_grid":
        return generate_tuple_grid_task(
            min_val=min_val, max_val=max_val, rng=rng
        )
    raise ValueError(f"unknown task type: {task_type!r}")


def _generate_unique_for_type(
    task_type: str,
    n: int,
    rng: random.Random,
    seen_matrices: set[str],
    seen_tasks: set[str],
    *,
    min_val: int,
    max_val: int,
    max_attempts: int = 500_000,
) -> list[dict]:
    """Generate ``n`` unique 3×3 tasks of ``task_type`` avoiding seen matrices."""
    out: list[dict] = []
    attempts = 0
    step_i = 0
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could only generate {len(out)}/{n} unique {task_type} tasks "
                f"after {attempts} attempts (min={min_val}, max={max_val}). "
                f"Try widening the digit band."
            )
        step = None
        if task_type == "progression_plus_n":
            step = PROGRESSION_STEPS[step_i % len(PROGRESSION_STEPS)]
            step_i += 1
        cand = _normalize_task(
            generate_candidate(
                task_type, rng, min_val=min_val, max_val=max_val, step=step
            )
        )
        mf = matrix_fingerprint(cand)
        tf = task_fingerprint(cand)
        if mf in seen_matrices or tf in seen_tasks:
            continue
        # For magnitude types, keep the answer inside the requested band.
        if task_type in MAGNITUDE_TYPES:
            mag = task_answer_magnitude(cand)
            if mag is None or mag < min_val or mag > max_val:
                continue
        seen_matrices.add(mf)
        seen_tasks.add(tf)
        out.append(cand)
    return out


def _even_half(n: int) -> tuple[int, int]:
    """Split ``n`` into (small_count, large_count) as evenly as possible."""
    small = n // 2
    large = n - small
    return small, large


def _allocate_band_pool(
    pool: list[dict],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    need = n_train + n_val + n_test
    if len(pool) != need:
        raise RuntimeError(f"Expected pool of {need}, got {len(pool)}")
    rng.shuffle(pool)
    test = pool[:n_test]
    val = pool[n_test : n_test + n_val]
    train = pool[n_test + n_val :]
    return train, val, test


def task_to_sft_row(task: dict) -> dict:
    """0-shot completion JSONL row."""
    prompt = matrix_to_completion_query(task["matrix"])
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


def write_json_tasks(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"tasks": tasks}, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_pilot(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample ``n`` rows stratified across task types (as evenly as possible)."""
    if n <= 0:
        return []
    if n >= len(rows):
        return list(rows)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["task_type"])].append(row)

    for tt in by_type:
        rng.shuffle(by_type[tt])

    types = [tt for tt in TASK_TYPE_CYCLE if by_type.get(tt)]
    base = n // len(types)
    rem = n % len(types)
    quotas = {tt: base + (1 if i < rem else 0) for i, tt in enumerate(types)}

    selected: list[dict] = []
    leftovers: list[dict] = []
    for tt in types:
        bucket = by_type[tt]
        take = min(quotas[tt], len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def _icl_seen() -> tuple[set[str], set[str]]:
    seen_matrices: set[str] = set()
    seen_tasks: set[str] = set()
    for examples in IN_CONTEXT_EXAMPLES.values():
        for ex in examples:
            seen_matrices.add(matrix_fingerprint({"matrix": ex["matrix"]}))
    return seen_matrices, seen_tasks


def _filter_legacy(tasks: list[dict]) -> list[dict]:
    """Keep original 7-type items so re-runs do not duplicate challenge types."""
    return [t for t in tasks if t["task_type"] in LEGACY_TASK_TYPE_CYCLE]


def _seed_seen_from_tasks(
    *task_lists: list[dict],
) -> tuple[set[str], set[str]]:
    seen_matrices, seen_tasks = _icl_seen()
    for tasks in task_lists:
        for t in tasks:
            seen_matrices.add(matrix_fingerprint(t))
            seen_tasks.add(task_fingerprint(t))
    return seen_matrices, seen_tasks


def generate_splits(
    *,
    out_dir: Path,
    repo_tasks_json: Path,
    seed: int,
    pilot_sizes: tuple[int, ...],
    min_val: int,
    max_val: int,
    small_max: int,
    sync_repo_tasks: bool,
    challenge_json: Path | None = None,
) -> None:
    if not (min_val <= small_max < max_val):
        raise SystemExit(
            f"Need min_val <= small_max < max_val; got "
            f"min={min_val}, small_max={small_max}, max={max_val}"
        )

    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"
    test_path = out_dir / "test.json"
    if not (train_path.is_file() and val_path.is_file() and test_path.is_file()):
        raise SystemExit(
            f"Expected existing 7-type splits at {train_path}, {val_path}, "
            f"{test_path}; generate those first before extending with challenge types."
        )

    challenge_path = challenge_json or CHALLENGE_TASKS_JSON
    if not challenge_path.is_file():
        raise SystemExit(f"Missing challenge eval file: {challenge_path}")

    legacy_train = _filter_legacy(_load_tasks(train_path))
    legacy_val = _filter_legacy(_load_tasks(val_path))
    legacy_test = _filter_legacy(_load_tasks(test_path))
    challenge_test = [_normalize_task(t) for t in _load_tasks(challenge_path)]
    if len(challenge_test) != TEST_PER_TYPE * len(CHALLENGE_TYPE_CYCLE):
        raise SystemExit(
            f"{challenge_path} should have "
            f"{TEST_PER_TYPE * len(CHALLENGE_TYPE_CYCLE)} tasks, "
            f"got {len(challenge_test)}"
        )
    chal_counts = Counter(t["task_type"] for t in challenge_test)
    for tt in CHALLENGE_TYPE_CYCLE:
        if chal_counts.get(tt, 0) != TEST_PER_TYPE:
            raise SystemExit(
                f"{challenge_path}: expected {TEST_PER_TYPE} {tt}, "
                f"got {chal_counts.get(tt, 0)}"
            )

    train_s, train_l = _even_half(TRAIN_PER_TYPE)
    val_s, val_l = _even_half(VAL_PER_TYPE)
    small_need = train_s + val_s
    large_need = train_l + val_l

    small_lo, small_hi = min_val, small_max
    large_lo, large_hi = small_max + 1, max_val

    seen_matrices, seen_tasks = _seed_seen_from_tasks(
        legacy_train, legacy_val, legacy_test, challenge_test
    )
    rng = random.Random(seed)

    extra_train: list[dict] = []
    extra_val: list[dict] = []

    for task_type in CHALLENGE_TYPE_CYCLE:
        small_pool = _generate_unique_for_type(
            task_type,
            small_need,
            rng,
            seen_matrices,
            seen_tasks,
            min_val=small_lo,
            max_val=small_hi,
        )
        large_pool = _generate_unique_for_type(
            task_type,
            large_need,
            rng,
            seen_matrices,
            seen_tasks,
            min_val=large_lo,
            max_val=large_hi,
        )
        tr_s, va_s, _te_s = _allocate_band_pool(
            small_pool,
            n_train=train_s,
            n_val=val_s,
            n_test=0,
            rng=rng,
        )
        tr_l, va_l, _te_l = _allocate_band_pool(
            large_pool,
            n_train=train_l,
            n_val=val_l,
            n_test=0,
            rng=rng,
        )
        extra_train.extend(tr_s + tr_l)
        extra_val.extend(va_s + va_l)

    train_tasks = interleave_tasks_by_type(legacy_train + extra_train)
    val_tasks = interleave_tasks_by_type(legacy_val + extra_val)
    test_tasks = interleave_tasks_by_type(legacy_test + challenge_test)

    # Sanity: disjoint matrices.
    sets = {
        "train": {matrix_fingerprint(t) for t in train_tasks},
        "val": {matrix_fingerprint(t) for t in val_tasks},
        "test": {matrix_fingerprint(t) for t in test_tasks},
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise SystemExit(f"Internal error: {a} ∩ {b} = {len(overlap)} matrices")

    out_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = out_dir / "sft"
    pilots_dir = sft_dir / "pilots"

    write_json_tasks(out_dir / "train.json", train_tasks)
    write_json_tasks(out_dir / "val.json", val_tasks)
    write_json_tasks(out_dir / "test.json", test_tasks)
    if sync_repo_tasks:
        write_json_tasks(repo_tasks_json, test_tasks)

    train_rows = [task_to_sft_row(t) for t in train_tasks]
    val_rows = [task_to_sft_row(t) for t in val_tasks]
    write_jsonl(sft_dir / "train.jsonl", train_rows)
    write_jsonl(sft_dir / "val.jsonl", val_rows)

    pilot_rng = random.Random(seed + 1)
    for n in pilot_sizes:
        pilot = stratified_pilot(train_rows, n, pilot_rng)
        write_jsonl(pilots_dir / f"train_n{n}.jsonl", pilot)

    print(f"Wrote train ({len(train_tasks)}) -> {out_dir / 'train.json'}")
    print(f"Wrote val   ({len(val_tasks)}) -> {out_dir / 'val.json'}")
    print(f"Wrote test  ({len(test_tasks)}) -> {out_dir / 'test.json'}")
    if sync_repo_tasks:
        print(f"Synced complete eval JSON -> {repo_tasks_json}")
    print(f"Wrote SFT train/val JSONL under {sft_dir}")
    print(f"Small band: [{small_lo}, {small_hi}]  Large band: [{large_lo}, {large_hi}]")
    print(f"Per-type quotas: train={TRAIN_PER_TYPE} val={VAL_PER_TYPE} test={TEST_PER_TYPE}")
    print(
        f"Challenge train/val magnitude types: "
        f"train {train_s}/{train_l} small/large, val {val_s}/{val_l} "
        f"(test frozen from challenge_tasks.json)"
    )
    for split_name, tasks in (
        ("train", train_tasks),
        ("val", val_tasks),
        ("test", test_tasks),
    ):
        bands = Counter(
            magnitude_band(t, small_max=small_max)
            for t in tasks
            if t["task_type"] in MAGNITUDE_TYPES
        )
        print(f"  {split_name} magnitude counts: {dict(bands)}")
    if pilot_sizes:
        print(f"Pilots: {', '.join(f'n={n}' for n in pilot_sizes)} -> {pilots_dir}")


def parse_pilot_sizes(raw: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(raw, tuple):
        return raw
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    sizes = tuple(int(p) for p in parts)
    if any(s <= 0 for s in sizes):
        raise argparse.ArgumentTypeError("pilot sizes must be positive integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/val/test Raven SFT data with small/large magnitudes "
            "evenly split across splits (disjoint matrices)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "data",
        help="Output directory (default: babylm_finetune/data)",
    )
    parser.add_argument(
        "--repo-tasks-json",
        type=Path,
        default=None,
        help="Also sync test set here for Modal eval (default: data/complete.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--pilot-sizes",
        type=parse_pilot_sizes,
        default=DEFAULT_PILOT_SIZES,
        help="Comma-separated stratified pilot sizes (default: 10,50,100)",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_VAL,
        dest="min_val",
        help=f"Minimum digit for magnitude types (default: {DEFAULT_MIN_VAL})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_VAL,
        dest="max_val",
        help=f"Maximum digit for magnitude types (default: {DEFAULT_MAX_VAL})",
    )
    parser.add_argument(
        "--small-max",
        type=int,
        default=DEFAULT_SMALL_MAX,
        help=(
            f"Inclusive upper bound of the small band "
            f"(default: {DEFAULT_SMALL_MAX}; large is small_max+1..max)"
        ),
    )
    parser.add_argument(
        "--no-sync-repo-tasks",
        action="store_true",
        help="Do not overwrite data/complete.json",
    )
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Do not run check_overlaps.py after generation",
    )
    args = parser.parse_args()

    if args.min_val > args.max_val:
        raise SystemExit("--min must be <= --max")

    # Backup previous test once.
    out_dir = args.out_dir.resolve()
    test_path = out_dir / "test.json"
    backup = out_dir / "test_pre_magnitude_balance_backup.json"
    if test_path.is_file() and not backup.is_file():
        shutil.copy2(test_path, backup)
        print(f"Backed up previous test -> {backup}")

    generate_splits(
        out_dir=out_dir,
        repo_tasks_json=(args.repo_tasks_json or COMPLETE_JSON).resolve(),
        seed=args.seed,
        pilot_sizes=tuple(args.pilot_sizes),
        min_val=args.min_val,
        max_val=args.max_val,
        small_max=args.small_max,
        sync_repo_tasks=not args.no_sync_repo_tasks,
    )

    if not args.skip_overlap_check:
        from check_overlaps import check_overlaps

        check_overlaps(out_dir, small_max=args.small_max)


if __name__ == "__main__":
    main()
