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
    "Qwen/Qwen3-14B",
)

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
    "Qwen/Qwen3-14B": "A100",
}

DEFAULT_GPU_TIER = "A10G"

# Models at or above this size (billions of params) use a shorter vLLM context window.
_LARGE_MODEL_PARAMS_B = 10.0
_LARGE_MODEL_MAX_LEN = 1024

_SIZE_RE = re.compile(
    r"(?:pythia-)?(\d+(?:\.\d+)?)([mb])-deduped|Qwen3-(\d+(?:\.\d+)?)B",
    re.I,
)


def gpu_tier_for_model(model_id: str) -> str:
    """Return Modal GPU tier (``T4``, ``A10G``, or ``A100``) for a HuggingFace model id."""
    return MODEL_GPU_TIER.get(model_id, DEFAULT_GPU_TIER)


def max_model_len_for_model(model_id: str, default: int = 2048) -> int:
    """Shorter context for ≥10B models to reduce KV cache pressure on vLLM."""
    params_b = parse_params_billions(model_id)
    if params_b is not None and params_b >= _LARGE_MODEL_PARAMS_B:
        return _LARGE_MODEL_MAX_LEN
    return default


def model_family(model_id: str) -> str:
    if "pythia" in model_id.lower():
        return "pythia"
    if "qwen" in model_id.lower():
        return "qwen3"
    return "other"


def parse_params_billions(model_id: str) -> float | None:
    """Parse parameter count in billions from a HF model id, if recognized."""
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


def resolve_models_arg(models: str) -> list[str]:
    """Expand ``--models`` CLI value to a list of HuggingFace model ids.

    - ``sweep`` → ``SCALING_SWEEP_MODELS``
    - ``pythia`` / ``qwen3`` → family subset
    - comma-separated HF ids → explicit list
    """
    key = models.strip().lower()
    if key == "sweep":
        return list(SCALING_SWEEP_MODELS)
    if key == "pythia":
        return list(PYTHIA_SCALING_MODELS)
    if key == "qwen3":
        return list(QWEN3_SCALING_MODELS)
    return [m.strip() for m in models.split(",") if m.strip()]
