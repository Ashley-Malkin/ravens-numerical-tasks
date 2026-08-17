#!/usr/bin/env python3
"""Assert train/val/test Raven splits have no matrix overlaps and expected counts.

Also requires splits to be disjoint from the fixed Raven ``IN_CONTEXT_EXAMPLES``
bank, from ICL-augmented split files
``oneICL_{train,val,test}.json`` / ``threeICL_{train,val,test}.json`` (when
present; per-task ``icl`` lists), and (for magnitude types) an even small/large
split within each of train / val / test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_sft_splits import (  # noqa: E402
    CHALLENGE_TYPES,
    DEFAULT_SMALL_MAX,
    DIGIT_TYPES,
    MAGNITUDE_TYPES,
    magnitude_band,
)
from ravens_numerical.generation.generator import (  # noqa: E402
    TASK_TYPE_CYCLE,
    matrix_fingerprint,
)
from ravens_numerical.prompts.prompts import IN_CONTEXT_EXAMPLES  # noqa: E402

EXPECTED = {
    "train": 1800,
    "val": 200,
    "test": 500,  # 50 per subtask (10 types)
}
EXPECTED_PER_TYPE = {
    "train": 180,
    "val": 20,
    "test": 50,
}

# Filename → (split name, expected ``n_icl_per_type``) when the file is present.
ICL_SPLIT_FILES: dict[str, tuple[str, int]] = {
    "oneICL_train.json": ("train", 1),
    "threeICL_train.json": ("train", 3),
    "oneICL_val.json": ("val", 1),
    "threeICL_val.json": ("val", 3),
    "oneICL_test.json": ("test", 1),
    "threeICL_test.json": ("test", 3),
}


def _load_payload(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def _load_tasks(path: Path) -> list[dict]:
    data = _load_payload(path)
    return list(data["tasks"])


def _matrix_set(tasks: list[dict]) -> set[str]:
    return {matrix_fingerprint(t) for t in tasks}


def _icl_bank_matrix_set(bank: dict) -> set[str]:
    fps: set[str] = set()
    for examples in bank.values():
        for ex in examples:
            fps.add(matrix_fingerprint({"matrix": ex["matrix"]}))
    return fps


def _icl_matrix_set() -> set[str]:
    """Matrix fingerprints from the fixed Raven ICL demonstration bank."""
    return _icl_bank_matrix_set(IN_CONTEXT_EXAMPLES)


def _even_half(n: int) -> tuple[int, int]:
    return n // 2, n - n // 2


def _check_icl_split_file(
    path: Path,
    *,
    expected_split: str,
    expected_n: int,
    split_sets: dict[str, set[str]],
    prompt_icl: set[str],
    errors: list[str],
) -> tuple[set[str], dict[str, list[str]]]:
    """Validate one ICL split JSON; return (all ICL mats, per-task ICL map)."""
    label = path.name
    data = _load_payload(path)
    tasks = list(data.get("tasks", []))

    n_meta = data.get("n_icl_per_type")
    if n_meta is not None and int(n_meta) != expected_n:
        errors.append(
            f"{label}: n_icl_per_type={n_meta!r}, expected {expected_n}"
        )

    # Tasks must match the canonical split (same matrices).
    split_fps = split_sets.get(expected_split)
    if split_fps is not None:
        file_fps = _matrix_set(tasks)
        if file_fps != split_fps:
            missing = split_fps - file_fps
            extra = file_fps - split_fps
            errors.append(
                f"{label}: tasks must match {expected_split}.json "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

    # Prefer per-task ``icl``; fall back to legacy top-level bank.
    per_task = any(isinstance(t.get("icl"), list) for t in tasks)
    icl_fps: set[str] = set()
    task_icl: dict[str, list[str]] = {}
    by_type: dict[str, list[str]] = defaultdict(list)

    if per_task:
        for i, t in enumerate(tasks):
            demos = t.get("icl")
            if not isinstance(demos, list):
                errors.append(f"{label}: task[{i}] missing per-task 'icl' list")
                continue
            if len(demos) != expected_n:
                errors.append(
                    f"{label}: task[{i}] icl has {len(demos)} demos, "
                    f"expected {expected_n}"
                )
            tt = t.get("task_type")
            demo_fps: list[str] = []
            for j, ex in enumerate(demos):
                if not isinstance(ex, dict) or "matrix" not in ex:
                    errors.append(
                        f"{label}: task[{i}].icl[{j}] missing matrix"
                    )
                    continue
                fp = matrix_fingerprint({"matrix": ex["matrix"]})
                demo_fps.append(fp)
                icl_fps.add(fp)
                if tt is not None:
                    by_type[str(tt)].append(fp)
            task_icl[matrix_fingerprint(t)] = demo_fps
    else:
        icl = data.get("icl")
        if not isinstance(icl, dict):
            errors.append(
                f"{label}: missing per-task 'icl' and top-level 'icl' bank"
            )
            return set(), {}
        for tt in TASK_TYPE_CYCLE:
            examples = icl.get(tt)
            if not isinstance(examples, list):
                errors.append(f"{label}: icl[{tt!r}] missing or not a list")
                continue
            if len(examples) != expected_n:
                errors.append(
                    f"{label}: icl[{tt!r}] has {len(examples)} demos, "
                    f"expected {expected_n}"
                )
            for ex in examples:
                fp = matrix_fingerprint({"matrix": ex["matrix"]})
                icl_fps.add(fp)
                by_type[tt].append(fp)

    n_demo_slots = sum(
        len(t.get("icl") or [])
        for t in tasks
        if isinstance(t.get("icl"), list)
    )
    if per_task and n_demo_slots and len(icl_fps) != n_demo_slots:
        errors.append(
            f"{label}: duplicate matrices within per-task ICL "
            f"({n_demo_slots - len(icl_fps)} dups)"
        )
    elif not per_task:
        n_legacy = sum(
            len(v) for v in (data.get("icl") or {}).values() if isinstance(v, list)
        )
        if n_legacy and len(icl_fps) != n_legacy:
            errors.append(f"{label}: duplicate matrices within ICL bank")

    # Within each task type, ICL demos must all be unique.
    for tt, fps in by_type.items():
        if len(fps) != len(set(fps)):
            errors.append(
                f"{label}: duplicate ICL matrices within type {tt!r} "
                f"({len(fps) - len(set(fps))} dups)"
            )

    for name, fps in split_sets.items():
        overlap = fps & icl_fps
        if overlap:
            errors.append(
                f"{name} ∩ {label} ICL: {len(overlap)} shared matrix fingerprint(s)"
            )

    overlap_prompt = icl_fps & prompt_icl
    if overlap_prompt:
        errors.append(
            f"{label} ICL ∩ IN_CONTEXT_EXAMPLES: "
            f"{len(overlap_prompt)} shared matrix fingerprint(s)"
        )

    return icl_fps, task_icl


def check_overlaps(
    data_dir: Path,
    *,
    strict_counts: bool = True,
    small_max: int = DEFAULT_SMALL_MAX,
) -> None:
    paths = {
        "train": data_dir / "train.json",
        "val": data_dir / "val.json",
        "test": data_dir / "test.json",
    }
    missing = [name for name, p in paths.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing split files under {data_dir}: {', '.join(missing)}")

    splits = {name: _load_tasks(path) for name, path in paths.items()}
    errors: list[str] = []

    for name, tasks in splits.items():
        fps = [matrix_fingerprint(t) for t in tasks]
        n_dups = len(fps) - len(set(fps))
        if n_dups:
            errors.append(f"{name}: duplicate matrices within split ({n_dups} dups)")

        if strict_counts and len(tasks) != EXPECTED[name]:
            errors.append(
                f"{name}: expected {EXPECTED[name]} tasks, got {len(tasks)}"
            )

        counts = Counter(t["task_type"] for t in tasks)
        for tt in TASK_TYPE_CYCLE:
            expected_n = EXPECTED_PER_TYPE[name]
            got = counts.get(tt, 0)
            if strict_counts and got != expected_n:
                errors.append(
                    f"{name}/{tt}: expected {expected_n} tasks, got {got}"
                )

        unknown = set(counts) - set(TASK_TYPE_CYCLE)
        if unknown:
            errors.append(f"{name}: unknown task types {sorted(unknown)}")

        # Even small/large within each magnitude type.
        # Challenge test items are a frozen suite (not 25/25 band-balanced).
        want_s, want_l = _even_half(EXPECTED_PER_TYPE[name])
        for tt in sorted(MAGNITUDE_TYPES):
            if name == "test" and tt in CHALLENGE_TYPES:
                continue
            band_counts = Counter(
                magnitude_band(t, small_max=small_max)
                for t in tasks
                if t["task_type"] == tt
            )
            got_s = band_counts.get("small", 0)
            got_l = band_counts.get("large", 0)
            if got_s != want_s or got_l != want_l:
                errors.append(
                    f"{name}/{tt}: expected small/large {want_s}/{want_l}, "
                    f"got {got_s}/{got_l}"
                )

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    sets = {name: _matrix_set(tasks) for name, tasks in splits.items()}
    for a, b in pairs:
        overlap = sets[a] & sets[b]
        if overlap:
            errors.append(
                f"{a} ∩ {b}: {len(overlap)} shared matrix fingerprint(s)"
            )

    icl_matrices = _icl_matrix_set()
    for name, fps in sets.items():
        overlap = fps & icl_matrices
        if overlap:
            errors.append(
                f"{name} ∩ ICL: {len(overlap)} shared matrix fingerprint(s) "
                f"with IN_CONTEXT_EXAMPLES"
            )

    # Optional ICL-augmented split files (one-shot / three-shot).
    icl_file_sets: dict[str, set[str]] = {}
    icl_task_maps: dict[str, dict[str, list[str]]] = {}
    for filename, (expected_split, expected_n) in ICL_SPLIT_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            continue
        fps, task_map = _check_icl_split_file(
            path,
            expected_split=expected_split,
            expected_n=expected_n,
            split_sets=sets,
            prompt_icl=icl_matrices,
            errors=errors,
        )
        icl_file_sets[filename] = fps
        icl_task_maps[filename] = task_map

    # Within each split: per-task one-shot ICL ⊆ three-shot ICL.
    for split in ("train", "val", "test"):
        one_name = f"oneICL_{split}.json"
        three_name = f"threeICL_{split}.json"
        one_map = icl_task_maps.get(one_name)
        three_map = icl_task_maps.get(three_name)
        if one_map is not None and three_map is not None and one_map and three_map:
            if set(one_map) != set(three_map):
                errors.append(
                    f"{one_name} / {three_name}: task sets differ "
                    f"(one={len(one_map)}, three={len(three_map)})"
                )
            mismatches = 0
            for tfp, one_demos in one_map.items():
                three_demos = three_map.get(tfp)
                if three_demos is None:
                    continue
                if one_demos != three_demos[: len(one_demos)]:
                    mismatches += 1
            if mismatches:
                errors.append(
                    f"{one_name} ICL is not a prefix of {three_name} ICL "
                    f"for {mismatches} task(s)"
                )
        else:
            # Legacy bank-level subset check.
            one = icl_file_sets.get(one_name)
            three = icl_file_sets.get(three_name)
            if one is not None and three is not None and one and three:
                if not one.issubset(three):
                    errors.append(
                        f"{one_name} ICL is not a subset of {three_name} ICL"
                    )

    # Distinct three-shot ICL across splits must be pairwise disjoint.
    three_banks = {
        split: icl_file_sets[f"threeICL_{split}.json"]
        for split in ("train", "val", "test")
        if f"threeICL_{split}.json" in icl_file_sets
        and icl_file_sets[f"threeICL_{split}.json"]
    }
    three_names = list(three_banks.keys())
    for i, a in enumerate(three_names):
        for b in three_names[i + 1 :]:
            overlap = three_banks[a] & three_banks[b]
            if overlap:
                errors.append(
                    f"threeICL_{a}.json ICL ∩ threeICL_{b}.json ICL: "
                    f"{len(overlap)} shared matrix fingerprint(s)"
                )

    if errors:
        print("OVERLAP CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Overlap check passed for {data_dir}")
    print(f"  ICL bank: {len(icl_matrices)} unique matrices checked")
    for filename, fps in icl_file_sets.items():
        print(f"  {filename}: {len(fps)} unique ICL matrices checked")
    print(
        f"  Magnitude types {sorted(MAGNITUDE_TYPES)} evenly split "
        f"(small ≤ {small_max}; digit types {sorted(DIGIT_TYPES)} exempt)"
    )
    for name, tasks in splits.items():
        print(
            f"  {name}: {len(tasks)} tasks, "
            f"unique_matrices={len(_matrix_set(tasks))}, "
            f"types={dict(Counter(t['task_type'] for t in tasks))}"
        )
        bands = Counter(
            magnitude_band(t, small_max=small_max)
            for t in tasks
            if t["task_type"] in MAGNITUDE_TYPES
        )
        print(f"    magnitude: {dict(bands)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Raven SFT train/val/test have no matrix overlaps "
            "with each other, IN_CONTEXT_EXAMPLES, or per-task ICL in "
            "oneICL_/threeICL_{train,val,test}.json, and even "
            "small/large magnitude splits."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "data",
        help="Directory containing train.json, val.json, test.json",
    )
    parser.add_argument(
        "--small-max",
        type=int,
        default=DEFAULT_SMALL_MAX,
        help=f"Inclusive small-band upper bound (default: {DEFAULT_SMALL_MAX})",
    )
    args = parser.parse_args()
    check_overlaps(args.data_dir.resolve(), small_max=args.small_max)


if __name__ == "__main__":
    main()
