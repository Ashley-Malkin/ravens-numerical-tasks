"""Resolve train/val JSONL paths, applying task-type filters when requested."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from babylm_finetune.training.config import TrainConfig
from babylm_finetune.training.task_filter import (
    ALL_TYPES,
    materialize_run_view,
    resolve_include_types,
)


def resolve_data_files(cfg: TrainConfig) -> tuple[Path, Path, dict[str, Any]]:
    """Return (train_path, val_path, type_manifest).

    If include/holdout types are set, materialize a filtered run view under
    ``data/sft/runs/<run_name>/``. Otherwise use ``cfg.train_file`` / ``val_file``.
    """
    include = list(cfg.include_types)
    holdout = list(cfg.holdout_types)

    if include or holdout:
        include_list, holdout_list = resolve_include_types(
            include_types=include or None,
            holdout_types=holdout or None,
        )
        type_manifest = materialize_run_view(
            run_name=cfg.run_name,
            include=include_list,
            holdout=holdout_list,
            train_path=Path(cfg.train_file),
            val_path=Path(cfg.val_file),
        )
        return (
            Path(type_manifest["train_file"]),
            Path(type_manifest["val_file"]),
            type_manifest,
        )

    include_all, holdout_all = resolve_include_types()
    train_path = Path(cfg.train_file).resolve()
    val_path = Path(cfg.val_file).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    type_manifest = {
        "run_name": cfg.run_name,
        "include_types": include_all,
        "holdout_types": holdout_all,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "test_policy": cfg.test_policy,
        "all_types": list(ALL_TYPES),
    }
    return train_path, val_path, type_manifest
