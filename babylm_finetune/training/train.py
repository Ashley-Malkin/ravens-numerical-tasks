"""TRL SFTTrainer entry for Raven completion-only BabyLM finetuning."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

from babylm_finetune.training.config import TrainConfig, model_tag_from_id
from babylm_finetune.training.data import resolve_data_files


class _Tee(TextIO):
    """Write to both a log file and the original stream."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, data: str) -> int:  # type: ignore[override]
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return self._stream.isatty()


def make_run_id(cfg: TrainConfig, *, utc: datetime | None = None) -> str:
    if cfg.run_id:
        return cfg.run_id
    ts = (utc or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    tag = model_tag_from_id(cfg.model_name_or_path)
    return f"{tag}__{cfg.run_name}__{ts}"


def run_sft_train(cfg: TrainConfig) -> dict[str, Any]:
    """Run completion-only SFT; write artifacts under ``cfg.output_dir``."""
    # Heavy imports deferred so ``--help`` / config resolve work without GPU deps.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from ravens_numerical.models.registry import (
        is_childes_model,
        is_os_model,
        is_td_model,
        is_ts_model,
        is_wiki_model,
        normalize_childes_model_id,
        normalize_os_model_id,
        normalize_td_model_id,
        normalize_ts_model_id,
        normalize_wiki_model_id,
        resolve_childes_local_path,
        resolve_os_local_path,
        resolve_td_local_path,
        resolve_ts_local_path,
        resolve_wiki_local_path,
    )

    # Keep logical Hub+subfolder id in manifests; load weights from a local rung dir.
    logical_model_id = cfg.model_name_or_path
    load_model_id = cfg.model_name_or_path
    if is_childes_model(cfg.model_name_or_path):
        logical_model_id = normalize_childes_model_id(cfg.model_name_or_path)
        load_model_id = resolve_childes_local_path(cfg.model_name_or_path)
        cfg.model_name_or_path = logical_model_id
    elif is_td_model(cfg.model_name_or_path):
        logical_model_id = normalize_td_model_id(cfg.model_name_or_path)
        load_model_id = resolve_td_local_path(cfg.model_name_or_path)
        cfg.model_name_or_path = logical_model_id
    elif is_os_model(cfg.model_name_or_path):
        logical_model_id = normalize_os_model_id(cfg.model_name_or_path)
        load_model_id = resolve_os_local_path(cfg.model_name_or_path)
        cfg.model_name_or_path = logical_model_id
    elif is_wiki_model(cfg.model_name_or_path):
        logical_model_id = normalize_wiki_model_id(cfg.model_name_or_path)
        load_model_id = resolve_wiki_local_path(cfg.model_name_or_path)
        cfg.model_name_or_path = logical_model_id
    elif is_ts_model(cfg.model_name_or_path):
        logical_model_id = normalize_ts_model_id(cfg.model_name_or_path)
        load_model_id = resolve_ts_local_path(cfg.model_name_or_path)
        cfg.model_name_or_path = logical_model_id

    run_id = make_run_id(cfg)
    output_dir = Path(cfg.output_dir)
    if output_dir.name != run_id and cfg.run_id is None:
        # When caller passes a parent dir, nest under run_id.
        if not str(output_dir).endswith(run_id):
            output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, type_manifest = resolve_data_files(cfg)

    log_path = output_dir / "train.log"
    tb_dir = output_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_name": cfg.run_name,
        "model_name_or_path": cfg.model_name_or_path,
        "include_types": type_manifest.get("include_types", []),
        "holdout_types": type_manifest.get("holdout_types", []),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "learning_rate": cfg.learning_rate,
        "seed": cfg.seed,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "test_policy": cfg.test_policy,
        "checkpoint_uri": str(output_dir),
        "completion_only_loss": cfg.completion_only_loss,
    }

    resolved = cfg.to_dict()
    resolved["run_id"] = run_id
    resolved["output_dir"] = str(output_dir)
    resolved["train_file"] = str(train_path)
    resolved["val_file"] = str(val_path)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(output_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    with open(log_path, "w", buffering=1) as log_f:
        tee_out = _Tee(sys.stdout, log_f)
        tee_err = _Tee(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"run_id={run_id}")
            print(f"output_dir={output_dir}")
            print(f"model={logical_model_id}")
            if load_model_id != logical_model_id:
                print(f"load_path={load_model_id}")
            print(f"train={train_path} val={val_path}")
            print(
                f"include={manifest['include_types']} "
                f"holdout={manifest['holdout_types']}"
            )

            tokenizer = AutoTokenizer.from_pretrained(load_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                load_model_id, torch_dtype="auto"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id

            dataset = load_dataset(
                "json",
                data_files={
                    "train": str(train_path),
                    "validation": str(val_path),
                },
            )

            sft_args = SFTConfig(
                output_dir=str(output_dir),
                learning_rate=cfg.learning_rate,
                num_train_epochs=cfg.num_train_epochs,
                max_steps=cfg.max_steps,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                warmup_ratio=cfg.warmup_ratio,
                weight_decay=cfg.weight_decay,
                lr_scheduler_type=cfg.lr_scheduler_type,
                seed=cfg.seed,
                logging_strategy="steps",
                logging_steps=cfg.logging_steps,
                logging_dir=str(tb_dir),
                report_to=["tensorboard"],
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                max_length=cfg.max_seq_length,
                completion_only_loss=cfg.completion_only_loss,
            )

            trainer = SFTTrainer(
                model=model,
                args=sft_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=tokenizer,
            )
            train_result = trainer.train()
            metrics = train_result.metrics if train_result is not None else {}
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            trainer.save_state()

            print(f"Saved model + tokenizer -> {output_dir}")
            print(f"metrics={metrics}")

    summary = {
        "run_id": run_id,
        "checkpoint_path": str(output_dir),
        "manifest": manifest,
        "metrics": dict(metrics) if metrics else {},
    }
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    return summary
