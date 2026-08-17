"""Train configuration for BabyLM Raven SFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "train.jsonl"
DEFAULT_VAL = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "val.jsonl"


@dataclass
class TrainConfig:
    model_name_or_path: str = "BabyLM-community/babylm-baseline-10m-gpt2"
    train_file: str = str(DEFAULT_TRAIN)
    val_file: str = str(DEFAULT_VAL)
    output_dir: str = str(REPO_ROOT / "babylm_finetune" / "outputs" / "local")
    run_name: str = "all_types"
    run_id: str | None = None

    include_types: list[str] = field(default_factory=list)
    holdout_types: list[str] = field(default_factory=list)

    learning_rate: float = 1.0e-4
    num_train_epochs: float = 5.0
    max_steps: int = -1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 512
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    logging_steps: int = 10
    completion_only_loss: bool = True
    mlm_probability: float = 0.15
    test_policy: str = "full_test_json_all_types"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_train_config(
    config_path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load YAML config and apply non-None overrides."""
    data: dict[str, Any] = {}
    if config_path is not None:
        data.update(_load_yaml(config_path.resolve()))
        # Allow composing with a base via ``extends:``
        extends = data.pop("extends", None)
        if extends:
            base_path = (config_path.parent / extends).resolve()
            base = _load_yaml(base_path)
            base.update(data)
            data = base

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            data[key] = value

    # Coerce list fields from comma-strings if needed
    for list_key in ("include_types", "holdout_types"):
        if list_key in data and isinstance(data[list_key], str):
            data[list_key] = [p.strip() for p in data[list_key].split(",") if p.strip()]

    valid = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in data.items() if k in valid}
    return TrainConfig(**filtered)


def model_tag_from_id(model_name_or_path: str) -> str:
    """Short safe tag for run_id (e.g. babylm-10m-gpt2, childes-seed42-rung1M)."""
    from ravens_numerical.models.registry import (
        childes_model_tag,
        miniberta_model_tag,
        os_model_tag,
        td_model_tag,
        ts_model_tag,
        wiki_model_tag,
    )

    childes = childes_model_tag(model_name_or_path)
    if childes is not None:
        return childes
    td = td_model_tag(model_name_or_path)
    if td is not None:
        return td
    os_tag = os_model_tag(model_name_or_path)
    if os_tag is not None:
        return os_tag
    wiki = wiki_model_tag(model_name_or_path)
    if wiki is not None:
        return wiki
    ts = ts_model_tag(model_name_or_path)
    if ts is not None:
        return ts
    mb = miniberta_model_tag(model_name_or_path)
    if mb is not None:
        return mb
    name = model_name_or_path.rstrip("/").split("/")[-1]
    name = name.replace("babylm-baseline-", "babylm-")
    return name.replace("/", "_").replace(":", "_")
