"""Known SFT ``--train-file`` / ``--val-file`` aliases and path resolution."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SFT_DIR = "babylm_finetune/data/sft"
PILOTS_DIR = f"{SFT_DIR}/pilots"

# Short names accepted by ``--train-file`` (also accepts full repo-relative paths).
TRAIN_FILE_ALIASES: dict[str, str] = {
    "train": f"{SFT_DIR}/train.jsonl",
    "oneICL": f"{SFT_DIR}/oneICL_train.jsonl",
    "oneICL_train": f"{SFT_DIR}/oneICL_train.jsonl",
    "threeICL": f"{SFT_DIR}/threeICL_train.jsonl",
    "threeICL_train": f"{SFT_DIR}/threeICL_train.jsonl",
    "train_n10": f"{PILOTS_DIR}/train_n10.jsonl",
    "train_n50": f"{PILOTS_DIR}/train_n50.jsonl",
    "train_n100": f"{PILOTS_DIR}/train_n100.jsonl",
}

VAL_FILE_ALIASES: dict[str, str] = {
    "val": f"{SFT_DIR}/val.jsonl",
    "oneICL": f"{SFT_DIR}/oneICL_val.jsonl",
    "oneICL_val": f"{SFT_DIR}/oneICL_val.jsonl",
    "threeICL": f"{SFT_DIR}/threeICL_val.jsonl",
    "threeICL_val": f"{SFT_DIR}/threeICL_val.jsonl",
}

# When ``--train-file`` uses an ICL alias and ``--val-file`` is omitted, pair
# with the matching ICL val JSONL (overrides the YAML zero-shot default).
_DEFAULT_VAL_ALIAS_FOR_TRAIN_ALIAS: dict[str, str] = {
    "oneICL": "oneICL",
    "oneICL_train": "oneICL",
    "threeICL": "threeICL",
    "threeICL_train": "threeICL",
}

_DEFAULT_VAL_REL_FOR_TRAIN_NAME: dict[str, str] = {
    "oneICL_train.jsonl": f"{SFT_DIR}/oneICL_val.jsonl",
    "threeICL_train.jsonl": f"{SFT_DIR}/threeICL_val.jsonl",
}


def train_file_help() -> str:
    aliases = ", ".join(sorted(TRAIN_FILE_ALIASES))
    return (
        "Training JSONL path (repo-relative or absolute), or alias: "
        f"{aliases}"
    )


def val_file_help() -> str:
    aliases = ", ".join(sorted(VAL_FILE_ALIASES))
    return (
        "Validation JSONL path (repo-relative or absolute), or alias: "
        f"{aliases}. "
        "If omitted with an ICL train alias/path, the matching ICL val is used."
    )


def resolve_data_file_arg(
    raw: str | None,
    *,
    aliases: dict[str, str],
) -> str | None:
    """Map an alias or path to a repo-relative (or absolute) string."""
    if raw is None:
        return None
    key = raw.strip()
    if key in aliases:
        return aliases[key]
    return key


def resolve_train_file_arg(raw: str | None) -> str | None:
    return resolve_data_file_arg(raw, aliases=TRAIN_FILE_ALIASES)


def resolve_val_file_arg(raw: str | None) -> str | None:
    return resolve_data_file_arg(raw, aliases=VAL_FILE_ALIASES)


def paired_val_file_for_train(
    train_raw: str | None,
    *,
    val_raw: str | None,
) -> str | None:
    """Return resolved val path when train implies an ICL val and val is omitted."""
    if val_raw is not None or train_raw is None:
        return resolve_val_file_arg(val_raw)

    key = train_raw.strip()
    if key in _DEFAULT_VAL_ALIAS_FOR_TRAIN_ALIAS:
        return resolve_val_file_arg(_DEFAULT_VAL_ALIAS_FOR_TRAIN_ALIAS[key])

    resolved_train = resolve_train_file_arg(key) or key
    name = Path(resolved_train).name
    paired = _DEFAULT_VAL_REL_FOR_TRAIN_NAME.get(name)
    if paired is not None:
        return paired
    return None
