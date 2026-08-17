#!/usr/bin/env python3
"""Local / in-container CLI for BabyLM Raven SFT (TRL completion-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babylm_finetune.training.config import (  # noqa: E402
    load_train_config,
    model_tag_from_id,
)
from babylm_finetune.training.task_filter import parse_types  # noqa: E402
from babylm_finetune.training.train import make_run_id, run_sft_train  # noqa: E402
from babylm_finetune.training.train_files import (  # noqa: E402
    paired_val_file_for_train,
    resolve_train_file_arg,
    train_file_help,
    val_file_help,
)


def _resolve_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Finetune BabyLM GPT-2 on Raven completion JSONL "
            "(completion-only loss; 0-shot or ICL). Prefer Modal via "
            "modal_sft.py for GPU runs."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML under babylm_finetune/configs/train/",
    )
    p.add_argument("--model", type=str, default=None, dest="model_name_or_path")
    p.add_argument(
        "--train-file",
        type=str,
        default=None,
        help=train_file_help(),
    )
    p.add_argument(
        "--val-file",
        type=str,
        default=None,
        help=val_file_help(),
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--include-types", type=str, default=None)
    p.add_argument("--holdout-types", type=str, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-train-epochs", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config + data paths and print JSON; do not train",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    include = parse_types(args.include_types) if args.include_types else None
    holdout = parse_types(args.holdout_types) if args.holdout_types else None

    resolved_train = resolve_train_file_arg(args.train_file)
    resolved_val = paired_val_file_for_train(
        args.train_file, val_raw=args.val_file
    )

    model_id = args.model_name_or_path
    if model_id is not None:
        from ravens_numerical.models.registry import (
            is_childes_model,
            normalize_childes_model_id,
        )

        if is_childes_model(model_id):
            model_id = normalize_childes_model_id(model_id)

    overrides = {
        "model_name_or_path": model_id,
        "train_file": _resolve_path(resolved_train),
        "val_file": _resolve_path(resolved_val),
        "output_dir": _resolve_path(args.output_dir),
        "run_name": args.run_name,
        "run_id": args.run_id,
        "include_types": include,
        "holdout_types": holdout,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    config_path = args.config.resolve() if args.config else None
    if config_path is None:
        config_path = (
            REPO_ROOT / "babylm_finetune" / "configs" / "train" / "sft_babylm_10m.yaml"
        )

    cfg = load_train_config(config_path, overrides=overrides)

    # Resolve relative paths from YAML against repo root
    for key in ("train_file", "val_file", "output_dir"):
        val = getattr(cfg, key)
        path = Path(val)
        if not path.is_absolute():
            setattr(cfg, key, str((REPO_ROOT / path).resolve()))

    if args.dry_run:
        from babylm_finetune.training.data import resolve_data_files

        run_id = make_run_id(cfg)
        train_path, val_path, type_manifest = resolve_data_files(cfg)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_tag": model_tag_from_id(cfg.model_name_or_path),
                    "config": cfg.to_dict(),
                    "train_file": str(train_path),
                    "val_file": str(val_path),
                    "type_manifest": type_manifest,
                },
                indent=2,
            )
        )
        return

    summary = run_sft_train(cfg)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
