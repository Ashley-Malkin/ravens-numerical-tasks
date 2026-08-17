"""Shared task-type include/holdout helpers for SFT data views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALL_TYPES = (
    "constancy",
    "constancy_row",
    "pattern",
    "pattern_tuple",
    "progression",
    "combine",
    "intersection",
    "distribution_of_three",
    "progression_plus_n",
    "tuple_grid",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_JSONL = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "train.jsonl"
DEFAULT_VAL_JSONL = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "val.jsonl"
DEFAULT_RUNS_DIR = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "runs"


def parse_types(raw: str | None) -> list[str]:
    if not raw:
        return []
    types = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [t for t in types if t not in ALL_TYPES]
    if unknown:
        raise ValueError(f"Unknown task type(s): {unknown}. Valid: {list(ALL_TYPES)}")
    seen: set[str] = set()
    out: list[str] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_include_types(
    *,
    include_types: list[str] | None = None,
    holdout_types: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    include_types = include_types or []
    holdout_types = holdout_types or []
    if include_types and holdout_types:
        raise ValueError("Pass only one of include_types or holdout_types")
    if not include_types and not holdout_types:
        include = list(ALL_TYPES)
        holdout: list[str] = []
    elif include_types:
        include = list(include_types)
        holdout = [t for t in ALL_TYPES if t not in set(include)]
    else:
        holdout = list(holdout_types)
        holdout_set = set(holdout)
        include = [t for t in ALL_TYPES if t not in holdout_set]
    if not include:
        raise ValueError("Resolved include set is empty; refuse empty train/val")
    return include, holdout


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_rows(rows: list[dict[str, Any]], include: set[str]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("task_type")) in include]


def materialize_run_view(
    *,
    run_name: str,
    include_types: list[str] | None = None,
    holdout_types: list[str] | None = None,
    include: list[str] | None = None,
    holdout: list[str] | None = None,
    train_path: Path | None = None,
    val_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Filter full SFT JSONL into ``data/sft/runs/<run_name>/`` and write manifest.

    Pass either pre-resolved ``include``/``holdout`` or raw
    ``include_types``/``holdout_types`` (not both pairs).
    """
    if include is not None or holdout is not None:
        if include is None or holdout is None:
            raise ValueError("Pass both include and holdout when pre-resolving")
        include_list, holdout_list = list(include), list(holdout)
        if not include_list:
            raise ValueError("Resolved include set is empty; refuse empty train/val")
    else:
        include_list, holdout_list = resolve_include_types(
            include_types=include_types,
            holdout_types=holdout_types,
        )
    train_src = (train_path or DEFAULT_TRAIN_JSONL).resolve()
    val_src = (val_path or DEFAULT_VAL_JSONL).resolve()
    if not train_src.is_file() or not val_src.is_file():
        raise FileNotFoundError(
            f"Missing train/val JSONL.\n  {train_src}\n  {val_src}\n"
            "Run generate_sft_splits.py first."
        )

    dest = (out_dir or (DEFAULT_RUNS_DIR / run_name)).resolve()
    train_rows = select_rows(read_jsonl(train_src), set(include_list))
    val_rows = select_rows(read_jsonl(val_src), set(include_list))
    write_jsonl(dest / "train.jsonl", train_rows)
    write_jsonl(dest / "val.jsonl", val_rows)

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "include_types": include_list,
        "holdout_types": holdout_list,
        "source_train": str(train_src),
        "source_val": str(val_src),
        "train_file": str(dest / "train.jsonl"),
        "val_file": str(dest / "val.jsonl"),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_policy": "full_test_json_all_types",
        "note": "Test remains the fixed full tasks.json; this never filters test.",
    }
    with open(dest / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest
