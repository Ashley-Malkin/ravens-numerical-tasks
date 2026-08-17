#!/usr/bin/env python3
"""Run BabyLM Raven SFT on Modal; persist checkpoints on a Volume by run_id.

Primary entrypoint (from repo root)::

    modal run babylm_finetune/scripts/modal_sft.py \\
      --config babylm_finetune/configs/train/sft_babylm_10m.yaml \\
      --train-file train_n50

    # ICL-augmented full train (1 or 3 demos/type in the prompt):
    modal run babylm_finetune/scripts/modal_sft.py \\
      --config babylm_finetune/configs/train/sft_babylm_10m.yaml \\
      --train-file oneICL

    modal run babylm_finetune/scripts/modal_sft.py \\
      --config babylm_finetune/configs/train/sft_babylm_10m.yaml \\
      --train-file threeICL

``--train-file`` accepts repo-relative JSONL paths or aliases (``train_n50``,
``oneICL``, ``threeICL``, …). See ``babylm_finetune.training.train_files``.

Finetuned models are stored at ``/checkpoints/<run_id>/`` on Volume
``ravens-babylm-sft``. Future eval should mount that Volume and pass
``/checkpoints/<run_id>`` as the model path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import modal

_SCRIPT_DIR = Path(__file__).resolve().parent
_BABYLM_FT_DIR = _SCRIPT_DIR.parent
REPO_ROOT = _BABYLM_FT_DIR.parent

# Ensure local imports work when Modal serializes this module.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ravens_numerical.paths import CONTAINER_RAVENS_ROOT, REPO_ROOT as _PATHS_REPO  # noqa: E402
from babylm_finetune.training.train_files import (  # noqa: E402
    paired_val_file_for_train,
    resolve_train_file_arg,
)

# Prefer paths.REPO_ROOT when available (src layout); fall back to script-relative.
REPO_ROOT = _PATHS_REPO

CONTAINER_SRC = f"{CONTAINER_RAVENS_ROOT}/src"
CONTAINER_BABYLM_FT = f"{CONTAINER_RAVENS_ROOT}/babylm_finetune"
CHECKPOINTS_ROOT = Path("/checkpoints")

IGNORE_COPY = [
    "**/.venv/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/artifacts/**",
    "**/*.egg-info/**",
    "**/node_modules/**",
    "**/.pytest_cache/**",
    "**/babylm_finetune/outputs/**",
    "**/models--*/**",
    "**/ravens-logs-full/**",
    "**/open-subtitles-models/**",
    "**/wiki-models/**",
    "**/tiny-stories-model/**",
]

app = modal.App("ravens-babylm-sft")

hf_cache_volume = modal.Volume.from_name("ravens-hf-cache", create_if_missing=True)
sft_volume = modal.Volume.from_name("ravens-babylm-sft", create_if_missing=True)
td_base_volume = modal.Volume.from_name("ravens-td-base", create_if_missing=True)
os_base_volume = modal.Volume.from_name("ravens-os-base", create_if_missing=True)
wiki_base_volume = modal.Volume.from_name("ravens-wiki-base", create_if_missing=True)
ts_base_volume = modal.Volume.from_name("ravens-ts-base", create_if_missing=True)

sft_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.46",
        "trl>=0.14",
        "datasets>=2.19",
        "accelerate>=0.30",
        "pyyaml>=6.0",
        "tensorboard>=2.14",
    )
    .env(
        {
            "PYTHONPATH": f"{CONTAINER_RAVENS_ROOT}:{CONTAINER_SRC}",
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=str(CONTAINER_RAVENS_ROOT),
        copy=True,
        ignore=IGNORE_COPY,
    )
)


def _container_path(repo_relative: str | None) -> str | None:
    if repo_relative is None:
        return None
    p = Path(repo_relative)
    if p.is_absolute():
        # Already absolute (e.g. inside container or local resolve) — map if under repo
        try:
            rel = p.resolve().relative_to(REPO_ROOT)
            return str(CONTAINER_RAVENS_ROOT / rel)
        except ValueError:
            return str(p)
    return str(CONTAINER_RAVENS_ROOT / p)


@app.function(
    image=sft_image,
    gpu="T4",
    timeout=6 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/checkpoints": sft_volume,
        "/td-base": td_base_volume,
        "/os-base": os_base_volume,
        "/wiki-base": wiki_base_volume,
        "/ts-base": ts_base_volume,
    },
)
def run_sft_remote(
    config_rel: str,
    model_name_or_path: Optional[str] = None,
    train_file_rel: Optional[str] = None,
    val_file_rel: Optional[str] = None,
    run_name: Optional[str] = None,
    include_types: Optional[str] = None,
    holdout_types: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_train_epochs: Optional[float] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    trainer: str = "causal",
) -> dict[str, Any]:
    """Train on Modal GPU; write ``/checkpoints/<run_id>/`` and commit the Volume.

    ``trainer`` is ``causal`` (TRL SFT / GPT-2) or ``mlm`` (HF MaskedLM / MiniBERTa).
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(CONTAINER_RAVENS_ROOT))

    from babylm_finetune.training.config import load_train_config, model_tag_from_id
    from babylm_finetune.training.task_filter import parse_types
    from babylm_finetune.training.train import make_run_id, run_sft_train

    trainer_key = (trainer or "causal").strip().lower()
    if trainer_key not in ("causal", "mlm"):
        raise ValueError(f"trainer must be 'causal' or 'mlm', got {trainer!r}")

    config_path = Path(_container_path(config_rel) or "")
    overrides: dict = {
        "model_name_or_path": model_name_or_path,
        "train_file": _container_path(train_file_rel),
        "val_file": _container_path(val_file_rel),
        "run_name": run_name,
        "include_types": parse_types(include_types) if include_types else None,
        "holdout_types": parse_types(holdout_types) if holdout_types else None,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        "seed": seed,
    }
    cfg = load_train_config(config_path, overrides=overrides)

    # Re-resolve relative paths against container repo root
    for key in ("train_file", "val_file"):
        val = getattr(cfg, key)
        path = Path(val)
        if not path.is_absolute():
            setattr(cfg, key, str(CONTAINER_RAVENS_ROOT / path))
        elif str(path).startswith(str(REPO_ROOT)):
            rel = path.relative_to(REPO_ROOT)
            setattr(cfg, key, str(CONTAINER_RAVENS_ROOT / rel))

    run_id = make_run_id(cfg)
    cfg.run_id = run_id
    cfg.output_dir = str(CHECKPOINTS_ROOT / run_id)

    print(
        f"Modal SFT: trainer={trainer_key} model={cfg.model_name_or_path} "
        f"run_id={run_id} tag={model_tag_from_id(cfg.model_name_or_path)}",
        flush=True,
    )
    if trainer_key == "mlm":
        from babylm_finetune.training.train_mlm import run_mlm_train

        summary = run_mlm_train(cfg)
    else:
        summary = run_sft_train(cfg)

    # Persist for future Modal eval jobs.
    sft_volume.commit()
    hf_cache_volume.commit()

    summary["checkpoint_path"] = str(CHECKPOINTS_ROOT / run_id)
    summary["volume"] = "ravens-babylm-sft"
    summary["future_eval_model_path"] = str(CHECKPOINTS_ROOT / run_id)
    summary["trainer"] = trainer_key
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


@app.local_entrypoint()
def main(
    config: str = "babylm_finetune/configs/train/sft_babylm_10m.yaml",
    model: Optional[str] = None,
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    run_name: Optional[str] = None,
    include_types: Optional[str] = None,
    holdout_types: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_train_epochs: Optional[float] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    trainer: str = "causal",
    mirror_manifest: bool = True,
) -> None:
    """Launch remote SFT and optionally mirror manifest locally.

    ``train_file`` / ``val_file`` accept JSONL paths or aliases
    (``oneICL``, ``threeICL``, ``train_n50``, …).

    ``trainer`` is ``causal`` (default) or ``mlm`` (MiniBERTa / RoBERTa).
    """
    resolved_train = resolve_train_file_arg(train_file)
    resolved_val = paired_val_file_for_train(train_file, val_raw=val_file)

    # Expand short family aliases (childes-1M, td-10M, miniberta-10M, …).
    resolved_model = model
    if model is not None:
        from ravens_numerical.models.registry import (
            is_childes_model,
            is_miniberta_model,
            is_os_model,
            is_td_model,
            is_ts_model,
            is_wiki_model,
            normalize_childes_model_id,
            normalize_miniberta_model_id,
            normalize_os_model_id,
            normalize_td_model_id,
            normalize_ts_model_id,
            normalize_wiki_model_id,
        )

        if is_childes_model(model):
            resolved_model = normalize_childes_model_id(model)
        elif is_td_model(model):
            resolved_model = normalize_td_model_id(model)
        elif is_os_model(model):
            resolved_model = normalize_os_model_id(model)
        elif is_wiki_model(model):
            resolved_model = normalize_wiki_model_id(model)
        elif is_ts_model(model):
            resolved_model = normalize_ts_model_id(model)
        elif is_miniberta_model(model):
            resolved_model = normalize_miniberta_model_id(model)

    summary = run_sft_remote.remote(
        config_rel=config,
        model_name_or_path=resolved_model,
        train_file_rel=resolved_train,
        val_file_rel=resolved_val,
        run_name=run_name,
        include_types=include_types,
        holdout_types=holdout_types,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        seed=seed,
        trainer=trainer,
    )

    run_id = summary["run_id"]
    print(f"\n=== SFT complete ===", flush=True)
    print(f"run_id: {run_id}", flush=True)
    print(f"Volume path (for future eval): {summary['future_eval_model_path']}", flush=True)
    print(
        "Mount Volume 'ravens-babylm-sft' at /checkpoints and pass "
        f"model_id={summary['future_eval_model_path']}",
        flush=True,
    )

    if mirror_manifest:
        local_out = REPO_ROOT / "babylm_finetune" / "outputs" / run_id
        local_out.mkdir(parents=True, exist_ok=True)
        (local_out / "manifest.json").write_text(
            json.dumps(summary.get("manifest", summary), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (local_out / "train_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Mirrored manifest -> {local_out}", flush=True)
