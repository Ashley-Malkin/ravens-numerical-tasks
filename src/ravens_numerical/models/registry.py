"""Default model IDs aligned with ``evaluate.py`` and ``baby_reasoning_eval``.

Use these with ``baby_reasoning/script/run``: ``--backend vllm`` + HF ids, or ``--backend ollama``
+ tags below.
"""

from __future__ import annotations

import re

# Same tags as ``evaluate.DEBUG_MODELS`` (Ollama).
QWEN3_OLLAMA_MODELS: tuple[str, ...] = (
    "qwen3:30b",
    "qwen3:8b",
    "qwen3:4b",
    "qwen3:1.7b",
    "qwen3:0.6b",
)

# Typical small Pythia checkpoint for vLLM (match baby-reasoning ``do_eval.sh`` style).
DEFAULT_PYTHIA_VLLM = "EleutherAI/pythia-70m-deduped"
DEFAULT_QWEN3_VLLM = "Qwen/Qwen3-8B"
QWEN3_8B_BASE = "Qwen/Qwen3-8B-Base"

PYTHIA_SCALING_MODELS: tuple[str, ...] = (
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
)

QWEN3_SCALING_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    QWEN3_8B_BASE,
    "Qwen/Qwen3-14B",
)

BABYLM_SCALING_MODELS: tuple[str, ...] = (
    "BabyLM-community/babylm-baseline-10m-gpt2",
    "BabyLM-community/babylm-baseline-100m-gpt2",
)

DEFAULT_BABYLM_VLLM = BABYLM_SCALING_MODELS[0]

_MINIBERTA_CORPUS_SIZES = ("1M", "10M", "100M", "1B")
_MINIBERTA_SEEDS = (1, 2, 3)

MINIBERTA_SCALING_MODELS: tuple[str, ...] = tuple(
    f"nyu-mll/roberta-{'med-small-' if size == '1M' else 'base-'}{size}-{seed}"
    for size in _MINIBERTA_CORPUS_SIZES
    for seed in _MINIBERTA_SEEDS
)

DEFAULT_MINIBERTA_VLLM = MINIBERTA_SCALING_MODELS[0]

# Full ladder for ``modal run ... --models sweep``.
SCALING_SWEEP_MODELS: tuple[str, ...] = PYTHIA_SCALING_MODELS + QWEN3_SCALING_MODELS

# Modal GPU tier per HF model id.
MODEL_GPU_TIER: dict[str, str] = {
    "EleutherAI/pythia-70m-deduped": "T4",
    "EleutherAI/pythia-160m-deduped": "T4",
    "EleutherAI/pythia-410m-deduped": "T4",
    "EleutherAI/pythia-1b-deduped": "T4",
    "EleutherAI/pythia-1.4b-deduped": "T4",
    "EleutherAI/pythia-2.8b-deduped": "A10G",
    "EleutherAI/pythia-6.9b-deduped": "A10G",
    "EleutherAI/pythia-12b-deduped": "A100",
    "Qwen/Qwen3-0.6B": "T4",
    "Qwen/Qwen3-1.7B": "T4",
    "Qwen/Qwen3-4B": "A10G",
    "Qwen/Qwen3-8B": "A10G",
    QWEN3_8B_BASE: "A10G",
    "Qwen/Qwen3-14B": "A100",
    "BabyLM-community/babylm-baseline-10m-gpt2": "T4",
    "BabyLM-community/babylm-baseline-100m-gpt2": "T4",
    **{mid: "T4" for mid in MINIBERTA_SCALING_MODELS},
}

DEFAULT_GPU_TIER = "A10G"

# Models at or above this size (billions of params) use a shorter vLLM context window.
_LARGE_MODEL_PARAMS_B = 10.0
_LARGE_MODEL_MAX_LEN = 1024

_SIZE_RE = re.compile(
    r"(?:pythia-)?(\d+(?:\.\d+)?)([mb])-deduped|Qwen3-(\d+(?:\.\d+)?)B",
    re.I,
)

_BABYLM_PARAMS_B = 0.124
# BabyLM baselines are GPT-2 (max_position_embeddings=1024).
_BABYLM_MAX_LEN = 1024

_BABYLM_CORPUS_RE = re.compile(r"babylm-baseline-(\d+)m-gpt2", re.I)

_MINIBERTA_MED_SMALL_PARAMS_B = 0.045
_MINIBERTA_BASE_PARAMS_B = 0.125
_MINIBERTA_MAX_LEN = 512

_MINIBERTA_CORPUS_RE = re.compile(
    r"roberta-(?:med-small-|base-)(\d+)([mMbB])-", re.I
)
_MINIBERTA_SEED_RE = re.compile(r"-(\d+)$")


def gpu_tier_for_model(model_id: str) -> str:
    """Return Modal GPU tier (``T4``, ``A10G``, or ``A100``) for a HuggingFace model id."""
    return MODEL_GPU_TIER.get(model_id, DEFAULT_GPU_TIER)


def max_model_len_for_model(model_id: str, default: int = 2048) -> int:
    """vLLM context length per model (respects architecture limits)."""
    if is_miniberta_model(model_id):
        return _MINIBERTA_MAX_LEN
    if "babylm" in model_id.lower():
        return _BABYLM_MAX_LEN
    params_b = parse_params_billions(model_id)
    if params_b is not None and params_b >= _LARGE_MODEL_PARAMS_B:
        return _LARGE_MODEL_MAX_LEN
    return default


def model_family(model_id: str) -> str:
    if is_miniberta_model(model_id):
        return "miniberta"
    if "babylm" in model_id.lower():
        return "babylm"
    if "pythia" in model_id.lower():
        return "pythia"
    if "qwen" in model_id.lower():
        return "qwen3"
    return "other"


def is_miniberta_model(model_id: str) -> bool:
    """True for nyu-mll MiniBERTa RoBERTa checkpoints (``roberta-base-*M-*`` / ``med-small``)."""
    mid = model_id.lower()
    return mid.startswith("nyu-mll/roberta-") and (
        "med-small-" in mid or re.search(r"roberta-base-\d+[mMbB]-", mid) is not None
    )


def is_pythia_model(model_id: str) -> bool:
    return model_family(model_id) == "pythia"


def is_qwen3_model(model_id: str) -> bool:
    return model_family(model_id) == "qwen3"


def is_qwen3_base_model(model_id: str) -> bool:
    """Pretrained-only Qwen3 checkpoints (``*-Base``), not post-trained instruct."""
    return is_qwen3_model(model_id) and "-base" in model_id.lower()


def is_qwen3_instruct_model(model_id: str) -> bool:
    """Post-trained Qwen3 instruct checkpoints (excludes ``*-Base``)."""
    return is_qwen3_model(model_id) and not is_qwen3_base_model(model_id)


def is_babylm_model(model_id: str) -> bool:
    return model_family(model_id) == "babylm"


def uses_completions_choice_only(model_id: str) -> bool:
    """Causal LMs using completions + guided JSON for instruction choice_only."""
    return (
        is_pythia_model(model_id)
        or is_babylm_model(model_id)
        or is_qwen3_base_model(model_id)
    )


def resolve_instruction_prompt_mode(mode: str) -> str:
    """Expand ``auto`` to ``choice_only`` (default for Modal instruction eval)."""
    if mode == "auto":
        return "choice_only"
    return mode


def parse_params_billions(model_id: str) -> float | None:
    """Parse parameter count in billions from a HF model id, if recognized."""
    if is_miniberta_model(model_id):
        if "med-small" in model_id.lower():
            return _MINIBERTA_MED_SMALL_PARAMS_B
        return _MINIBERTA_BASE_PARAMS_B
    if is_babylm_model(model_id):
        return _BABYLM_PARAMS_B
    m = _SIZE_RE.search(model_id)
    if not m:
        return None
    if m.group(3):
        return float(m.group(3))
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return value / 1000.0
    return value


def parse_training_corpus_millions(model_id: str) -> float | None:
    """Training corpus size in millions of tokens/words when encoded in the HF id."""
    m = _MINIBERTA_CORPUS_RE.search(model_id)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "b":
            return value * 1000.0
        return value
    m = _BABYLM_CORPUS_RE.search(model_id)
    if not m:
        return None
    return float(m.group(1))


def parse_miniberta_seed(model_id: str) -> int | None:
    """Checkpoint seed suffix (1, 2, or 3) for MiniBERTa ids."""
    if not is_miniberta_model(model_id):
        return None
    m = _MINIBERTA_SEED_RE.search(model_id.rstrip("/"))
    if not m:
        return None
    return int(m.group(1))


def resolve_models_arg(models: str) -> list[str]:
    """Expand ``--models`` CLI value to a list of HuggingFace model ids.

    - ``sweep`` → ``SCALING_SWEEP_MODELS``
    - ``pythia`` / ``qwen3`` / ``babylm`` / ``miniberta`` → family subset
    - comma-separated HF ids → explicit list
    """
    key = models.strip().lower()
    if key == "sweep":
        return list(SCALING_SWEEP_MODELS)
    if key == "pythia":
        return list(PYTHIA_SCALING_MODELS)
    if key == "qwen3":
        return list(QWEN3_SCALING_MODELS)
    if key == "babylm":
        return list(BABYLM_SCALING_MODELS)
    if key == "miniberta":
        return list(MINIBERTA_SCALING_MODELS)
    return [m.strip() for m in models.split(",") if m.strip()]
