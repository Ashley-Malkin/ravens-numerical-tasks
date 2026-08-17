#!/usr/bin/env python3
"""Filter full SFT train/val JSONL by task type for holdout finetune runs.

Test stays the fixed full complete.json and is never filtered by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babylm_finetune.training.task_filter import (  # noqa: E402
    DEFAULT_RUNS_DIR,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VAL_JSONL,
    materialize_run_view,
    parse_types,
    resolve_include_types,
)


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --config. Install pyyaml or pass CLI flags."
        ) from exc
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select task types for a finetune run's train/val JSONL. "
            "Never modifies the fixed test set."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_JSONL,
        help="Full SFT train JSONL",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_JSONL,
        help="Full SFT val JSONL",
    )
    parser.add_argument(
        "--include-types",
        type=str,
        default=None,
        help="Comma-separated task types to include in train/val",
    )
    parser.add_argument(
        "--holdout-types",
        type=str,
        default=None,
        help="Comma-separated task types to exclude from train/val",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output directory name under data/sft/runs/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory (overrides --run-name default path)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML with include_types/holdout_types and run_name",
    )
    args = parser.parse_args()

    include_raw = args.include_types
    holdout_raw = args.holdout_types
    run_name = args.run_name

    if args.config is not None:
        cfg = _load_yaml_config(args.config.resolve())
        if include_raw is None and "include_types" in cfg:
            val = cfg["include_types"]
            include_raw = ",".join(val) if isinstance(val, list) else str(val)
        if holdout_raw is None and "holdout_types" in cfg:
            val = cfg["holdout_types"]
            holdout_raw = ",".join(val) if isinstance(val, list) else str(val)
        if run_name is None and cfg.get("run_name"):
            run_name = str(cfg["run_name"])

    include, holdout = resolve_include_types(
        include_types=parse_types(include_raw) or None,
        holdout_types=parse_types(holdout_raw) or None,
    )

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
        run_name = run_name or out_dir.name
    else:
        if not run_name:
            raise SystemExit("Provide --run-name or --out-dir (or run_name in --config)")
        out_dir = (DEFAULT_RUNS_DIR / run_name).resolve()

    try:
        manifest = materialize_run_view(
            run_name=run_name,
            include=include,
            holdout=holdout,
            train_path=args.train.resolve(),
            val_path=args.val.resolve(),
            out_dir=out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote run views -> {out_dir}")
    print(f"  include: {manifest['include_types']}")
    print(f"  holdout: {manifest['holdout_types']}")
    print(f"  train={manifest['train_count']} val={manifest['val_count']}")
    print("  test: unchanged (fixed full tasks.json)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
