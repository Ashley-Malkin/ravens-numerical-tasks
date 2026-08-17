"""HF Trainer entry for MiniBERTa / RoBERTa MLM finetuning on Raven JSONL."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import yaml

from babylm_finetune.training.config import TrainConfig
from babylm_finetune.training.data import resolve_data_files
from babylm_finetune.training.train import _Tee, make_run_id


def run_mlm_train(cfg: TrainConfig) -> dict[str, Any]:
    """Run MLM finetuning on ``prompt + completion`` text; write under ``cfg.output_dir``."""
    from datasets import load_dataset
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from ravens_numerical.models.registry import normalize_miniberta_model_id

    logical_model_id = normalize_miniberta_model_id(cfg.model_name_or_path)
    load_model_id = logical_model_id
    cfg.model_name_or_path = logical_model_id

    run_id = make_run_id(cfg)
    output_dir = Path(cfg.output_dir)
    if output_dir.name != run_id and cfg.run_id is None:
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
        "trainer": "mlm",
        "mlm_probability": cfg.mlm_probability,
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
    }

    resolved = cfg.to_dict()
    resolved["run_id"] = run_id
    resolved["output_dir"] = str(output_dir)
    resolved["train_file"] = str(train_path)
    resolved["val_file"] = str(val_path)
    resolved["trainer"] = "mlm"

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
            print(f"trainer=mlm mlm_probability={cfg.mlm_probability}")
            print(f"train={train_path} val={val_path}")
            print(
                f"include={manifest['include_types']} "
                f"holdout={manifest['holdout_types']}"
            )

            tokenizer = AutoTokenizer.from_pretrained(load_model_id)
            model = AutoModelForMaskedLM.from_pretrained(load_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            raw = load_dataset(
                "json",
                data_files={
                    "train": str(train_path),
                    "validation": str(val_path),
                },
            )

            def _to_text(batch: dict[str, list]) -> dict[str, list]:
                texts = [
                    (p or "") + (c or "")
                    for p, c in zip(batch["prompt"], batch["completion"])
                ]
                return {"text": texts}

            text_ds = raw.map(
                _to_text,
                batched=True,
                remove_columns=raw["train"].column_names,
            )

            def _tokenize(batch: dict[str, list]) -> dict[str, list]:
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=cfg.max_seq_length,
                    padding=False,
                )

            tokenized = text_ds.map(
                _tokenize,
                batched=True,
                remove_columns=["text"],
            )

            collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=cfg.mlm_probability,
            )

            train_args = TrainingArguments(
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
            )

            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized["validation"],
                data_collator=collator,
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
