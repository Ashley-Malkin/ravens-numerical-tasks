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

# Training steps for ``--models pythia-checkpoints`` (HF revision branches).
PYTHIA_CHECKPOINT_STEPS: tuple[int, ...] = (
    64,
    512,
    1000,
    5000,
    10000,
    15000,
    20000,
    25000,
)

# Pythia trains with a uniform 2M-token batch (1024 × 2048) across model sizes.
PYTHIA_TOKENS_PER_STEP = 2_097_152
# Full ``main`` / step143000 checkpoints see ~300B tokens (deduped: ~1.5 epochs).
PYTHIA_FULL_TRAINING_TOKENS_B = 300.0

# Pythia sizes included in ``--models pythia-checkpoints`` (70M, 160M, 1B).
PYTHIA_CHECKPOINT_SCALING_MODELS: tuple[str, ...] = (
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-1b-deduped",
)

OLMO2_BASE_MODELS: tuple[str, ...] = (
    "allenai/OLMo-2-0425-1B",
    "allenai/OLMo2-7B-1124",
    "allenai/OLMo-2-13B-1124",
)

OLMO2_INSTRUCT_MODELS: tuple[str, ...] = (
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
)

OLMO2_MODELS: tuple[str, ...] = OLMO2_BASE_MODELS + OLMO2_INSTRUCT_MODELS

# OLMo 2 base sizes for ``--models olmo2-checkpoints`` (same as base ladder).
OLMO2_CHECKPOINT_SCALING_MODELS: tuple[str, ...] = OLMO2_BASE_MODELS

# Target tokens-viewed budgets (billions) for ``olmo2-checkpoints``.
OLMO2_CHECKPOINT_TOKEN_BUDGETS_B: tuple[int, ...] = (1, 21, 42, 49, 63)

# Stage-1 HF revisions per base model (token order). Shared budgets are
# ``OLMO2_CHECKPOINT_TOKEN_BUDGETS_B``; 7B/13B add early checkpoints (5B/10B)
# where published. Approximate labels: 7B 47B≈49B; 13B 0B≈1B, 17B≈21B,
# 51B≈49B, 59B≈63B.
OLMO2_CHECKPOINT_REVISIONS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "allenai/OLMo-2-0425-1B": (
        "stage1-step300-tokens1B",
        "stage1-step10000-tokens21B",
        "stage1-step20000-tokens42B",
        "stage1-step23100-tokens49B",
        "stage1-step30000-tokens63B",
    ),
    "allenai/OLMo2-7B-1124": (
        "stage1-step150-tokens1B",
        "stage1-step1000-tokens5B",
        "stage1-step2150-tokens10B",
        "stage1-step5000-tokens21B",
        "stage1-step10000-tokens42B",
        "stage1-step11000-tokens47B",
        "stage1-step15000-tokens63B",
    ),
    "allenai/OLMo-2-13B-1124": (
        "stage1-step0-tokens0B",
        "stage1-step1100-tokens10B",
        "stage1-step2000-tokens17B",
        "stage1-step5000-tokens42B",
        "stage1-step6000-tokens51B",
        "stage1-step7000-tokens59B",
    ),
}

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

# CHILDES GPT-2 developmental ladder (single Hub repo; checkpoints are subfolders).
CHILDES_LADDER_REPO = "mcxfrank/childes-gpt2-ladder"
CHILDES_LADDER_SEED = 42
CHILDES_LADDER_BUDGETS: tuple[str, ...] = ("1M", "5M", "12M", "24M")


def childes_ladder_model_id(
    budget: str,
    *,
    seed: int = CHILDES_LADDER_SEED,
) -> str:
    """Logical id: ``mcxfrank/childes-gpt2-ladder/development/seed{S}/rung{B}``."""
    b = budget.strip()
    if b.lower().endswith("m") and not b.endswith("M"):
        b = b[:-1] + "M"
    return f"{CHILDES_LADDER_REPO}/development/seed{seed}/rung{b}"


CHILDES_SCALING_MODELS: tuple[str, ...] = tuple(
    childes_ladder_model_id(b) for b in CHILDES_LADDER_BUDGETS
)

DEFAULT_CHILDES_VLLM = CHILDES_SCALING_MODELS[0]

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
    **{mid: "T4" for mid in CHILDES_SCALING_MODELS},
    "allenai/OLMo-2-0425-1B": "T4",
    "allenai/OLMo-2-0425-1B-Instruct": "T4",
    "allenai/OLMo2-7B-1124": "A10G",
    "allenai/OLMo-2-1124-7B-Instruct": "A10G",
    "allenai/OLMo-2-13B-1124": "A100",
    "allenai/OLMo-2-1124-13B-Instruct": "A100",
    **{mid: "T4" for mid in MINIBERTA_SCALING_MODELS},
}

DEFAULT_GPU_TIER = "A10G"

# Models at or above this size (billions of params) use a shorter vLLM context window.
_LARGE_MODEL_PARAMS_B = 10.0
_LARGE_MODEL_MAX_LEN = 1024

_SIZE_RE = re.compile(
    r"(?:pythia-)?(\d+(?:\.\d+)?)([mb])-deduped|Qwen3-(\d+(?:\.\d+)?)B|OLMo(?:-2)?-\d{4}-(\d+)B|OLMo2-(\d+)B-\d{4}|OLMo-2-(\d+)B-\d{4}",
    re.I,
)

_BABYLM_PARAMS_B = 0.124
# BabyLM baselines are GPT-2 (max_position_embeddings=1024).
_BABYLM_MAX_LEN = 1024

_BABYLM_CORPUS_RE = re.compile(r"babylm-baseline-(\d+)m-gpt2", re.I)

# CHILDES ladder: GPT-2-small (~125.8M), context 1024, custom 52k tokenizer.
_CHILDES_PARAMS_B = 0.126
_CHILDES_MAX_LEN = 1024
_CHILDES_SHORT_RE = re.compile(r"^childes-(\d+(?:\.\d+)?)[mM]$", re.I)
_CHILDES_SUBFOLDER_RE = re.compile(
    r"^development/seed(\d+)/rung(\d+(?:\.\d+)?)[mM]$",
    re.I,
)
_CHILDES_RUNG_CORPUS_RE = re.compile(r"rung(\d+(?:\.\d+)?)[mM]", re.I)

_MINIBERTA_MED_SMALL_PARAMS_B = 0.045
_MINIBERTA_BASE_PARAMS_B = 0.125
_MINIBERTA_MAX_LEN = 512

_MINIBERTA_CORPUS_RE = re.compile(
    r"roberta-(?:med-small-|base-)(\d+)([mMbB])-", re.I
)
_MINIBERTA_SEED_RE = re.compile(r"-(\d+)$")


_OLMO2_TOKENS_B_RE = re.compile(r"tokens(\d+)B", re.I)
_OLMO2_STEP_RE = re.compile(r"stage1-step(\d+)", re.I)


def pythia_checkpoint_revision(step: int) -> str:
    """HuggingFace branch name for a Pythia training step (e.g. ``step64``)."""
    return f"step{step}"


def pythia_step_from_revision(revision: str) -> int | None:
    """Parse ``stepN`` revision to training step ``N``."""
    if not revision.startswith("step") or not revision[4:].isdigit():
        return None
    return int(revision[4:])


def pythia_tokens_billions_at_step(step: int) -> float:
    """Tokens seen at a Pythia checkpoint step, in billions."""
    return step * PYTHIA_TOKENS_PER_STEP / 1e9


def format_pythia_checkpoint_model_id(base_model_id: str, step: int) -> str:
    """Display id for a Pythia checkpoint: ``{base}@step{N}``."""
    return f"{base_model_id}@{pythia_checkpoint_revision(step)}"


def _is_pythia_hf_revision(revision: str) -> bool:
    return revision.startswith("step") and revision[4:].isdigit()


def _is_olmo2_hf_revision(revision: str) -> bool:
    return revision.startswith("stage1-") and _OLMO2_TOKENS_B_RE.search(revision) is not None


def parse_checkpoint_model_id(model_id: str) -> tuple[str, str | None]:
    """Split ``base@revision`` into ``(base_model_id, revision)`` for known HF revisions.

    Accepts Pythia ``stepN`` and OLMo 2 ``stage1-step*-tokens*B`` suffixes.
    """
    if "@" not in model_id:
        return model_id, None
    base, revision = model_id.rsplit("@", 1)
    if _is_pythia_hf_revision(revision) or _is_olmo2_hf_revision(revision):
        return base, revision
    return model_id, None


def parse_pythia_checkpoint_model_id(
    model_id: str,
) -> tuple[str, str | None]:
    """Split ``base@stepN`` / ``base@stage1-...`` into ``(base, revision)``.

    Kept for callers; prefers any recognized checkpoint revision suffix.
    """
    return parse_checkpoint_model_id(model_id)


def base_model_id(model_id: str) -> str:
    """Strip a checkpoint suffix (``@stepN`` / ``@stage1-...``) when present."""
    base, _ = parse_checkpoint_model_id(model_id)
    return base


def is_pythia_checkpoint_model_id(model_id: str) -> bool:
    """True when ``model_id`` encodes a Pythia HF revision (``...@stepN``)."""
    _, revision = parse_checkpoint_model_id(model_id)
    return revision is not None and _is_pythia_hf_revision(revision)


def is_olmo2_checkpoint_model_id(model_id: str) -> bool:
    """True when ``model_id`` encodes an OLMo 2 stage-1 HF revision."""
    _, revision = parse_checkpoint_model_id(model_id)
    return revision is not None and _is_olmo2_hf_revision(revision)


def format_olmo2_checkpoint_model_id(base: str, revision: str) -> str:
    """Display id for an OLMo 2 checkpoint: ``{base}@{stage1-...}``."""
    return f"{base}@{revision}"


def olmo2_tokens_billions_from_revision(revision: str) -> float | None:
    """Parse ``tokensNB`` from an OLMo stage-1 revision name."""
    m = _OLMO2_TOKENS_B_RE.search(revision)
    if not m:
        return None
    return float(m.group(1))


def olmo2_step_from_revision(revision: str) -> int | None:
    """Parse training step from an OLMo stage-1 revision (e.g. ``stage1-step300-...``)."""
    m = _OLMO2_STEP_RE.search(revision)
    if not m:
        return None
    return int(m.group(1))


def checkpoint_step_from_model_id(model_id: str) -> int | None:
    """Training step encoded in a checkpoint model id, if any.

    Pythia ``...@stepN`` → ``N``; OLMo 2 ``...@stage1-stepN-...`` → ``N``.
    """
    _, revision = parse_checkpoint_model_id(model_id)
    if revision is None:
        return None
    if _is_pythia_hf_revision(revision):
        return pythia_step_from_revision(revision)
    if _is_olmo2_hf_revision(revision):
        return olmo2_step_from_revision(revision)
    return None


def pythia_checkpoint_model_ids() -> list[str]:
    """Pythia checkpoint sweep sizes × ``PYTHIA_CHECKPOINT_STEPS`` checkpoint ids."""
    return [
        format_pythia_checkpoint_model_id(base, step)
        for base in PYTHIA_CHECKPOINT_SCALING_MODELS
        for step in PYTHIA_CHECKPOINT_STEPS
    ]


def olmo2_checkpoint_model_ids() -> list[str]:
    """OLMo 2 base sizes × stage-1 revisions for ``OLMO2_CHECKPOINT_TOKEN_BUDGETS_B``."""
    return [
        format_olmo2_checkpoint_model_id(base, revision)
        for base in OLMO2_CHECKPOINT_SCALING_MODELS
        for revision in OLMO2_CHECKPOINT_REVISIONS_BY_MODEL[base]
    ]


def gpu_tier_for_model(model_id: str) -> str:
    """Return Modal GPU tier (``T4``, ``A10G``, or ``A100``) for a HuggingFace model id."""
    if is_sft_checkpoint_model(model_id):
        return "T4"
    if is_childes_model(model_id):
        return "T4"
    return MODEL_GPU_TIER.get(base_model_id(model_id), DEFAULT_GPU_TIER)


def max_model_len_for_model(model_id: str, default: int = 2048) -> int:
    """vLLM context length per model (respects architecture limits)."""
    model_id = base_model_id(model_id)
    if is_miniberta_model(model_id):
        return _MINIBERTA_MAX_LEN
    # Base ladder ids + SFT run_ids / Volume paths (``.../childes-seed42-rung…``).
    if is_childes_model(model_id) or "childes-" in model_id.lower():
        return _CHILDES_MAX_LEN
    if "babylm" in model_id.lower():
        return _BABYLM_MAX_LEN
    params_b = parse_params_billions(model_id)
    if params_b is not None and params_b >= _LARGE_MODEL_PARAMS_B:
        return _LARGE_MODEL_MAX_LEN
    return default


def model_family(model_id: str) -> str:
    model_id = base_model_id(model_id)
    if is_miniberta_model(model_id):
        return "miniberta"
    if is_childes_model(model_id) or "childes-" in model_id.lower():
        return "childes"
    if "babylm" in model_id.lower():
        return "babylm"
    if "pythia" in model_id.lower():
        return "pythia"
    if "olmo" in model_id.lower():
        return "olmo2"
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


def is_olmo2_model(model_id: str) -> bool:
    return model_family(model_id) == "olmo2"


def is_olmo2_instruct_model(model_id: str) -> bool:
    """Post-trained OLMo 2 instruct checkpoints (chat-tuned)."""
    return is_olmo2_model(model_id) and "instruct" in base_model_id(model_id).lower()


def is_olmo2_base_model(model_id: str) -> bool:
    """Pretrained-only OLMo 2 checkpoints (not instruct)."""
    return is_olmo2_model(model_id) and not is_olmo2_instruct_model(model_id)


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


def _normalize_childes_budget(raw: str) -> str:
    b = raw.strip()
    if b.lower().endswith("m"):
        return b[:-1] + "M"
    return b


def parse_childes_ladder_id(model_id: str) -> tuple[str, str] | None:
    """Parse a CHILDES ladder id into ``(repo_id, subfolder)``.

    Accepts short aliases (``childes-1M``) and full logical ids
    (``mcxfrank/childes-gpt2-ladder/development/seed42/rung1M``).
    """
    mid = model_id.strip().rstrip("/")
    short = _CHILDES_SHORT_RE.match(mid)
    if short:
        budget = _normalize_childes_budget(short.group(1) + "M")
        sub = f"development/seed{CHILDES_LADDER_SEED}/rung{budget}"
        return CHILDES_LADDER_REPO, sub

    prefix = f"{CHILDES_LADDER_REPO}/"
    if not mid.startswith(prefix):
        return None
    sub = mid[len(prefix) :]
    m = _CHILDES_SUBFOLDER_RE.match(sub)
    if not m:
        return None
    seed, budget = m.group(1), _normalize_childes_budget(m.group(2) + "M")
    return CHILDES_LADDER_REPO, f"development/seed{seed}/rung{budget}"


def normalize_childes_model_id(model_id: str) -> str:
    """Expand short CHILDES aliases to the full logical Hub+subfolder id."""
    parsed = parse_childes_ladder_id(model_id)
    if parsed is None:
        return model_id.strip().rstrip("/")
    repo, sub = parsed
    return f"{repo}/{sub}"


def is_childes_model(model_id: str) -> bool:
    """True for CHILDES ladder Hub/subfolder ids (not SFT run_ids)."""
    return parse_childes_ladder_id(model_id) is not None


def childes_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``childes-seed42-rung1M``, or ``None`` if not CHILDES."""
    parsed = parse_childes_ladder_id(model_id)
    if parsed is None:
        return None
    _, sub = parsed
    # development/seed42/rung1M → childes-seed42-rung1M
    parts = sub.split("/")
    if len(parts) >= 3:
        return f"childes-{parts[1]}-{parts[2]}"
    return f"childes-{parts[-1]}"


def resolve_childes_local_path(
    model_id: str,
    *,
    cache_dir: str | None = None,
) -> str:
    """Download a CHILDES rung subfolder and return its local directory path.

    Non-CHILDES ids are returned unchanged. Uses ``huggingface_hub.snapshot_download``
    with ``allow_patterns`` so only the requested rung is fetched.
    """
    from pathlib import Path

    parsed = parse_childes_ladder_id(model_id)
    if parsed is None:
        return model_id.strip().rstrip("/")
    repo, subfolder = parsed
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=repo,
        allow_patterns=[f"{subfolder}/**"],
        cache_dir=cache_dir,
    )
    local = Path(root) / subfolder
    if not local.is_dir():
        raise FileNotFoundError(
            f"CHILDES ladder subfolder not found after download: {local} "
            f"(repo={repo!r}, subfolder={subfolder!r})"
        )
    return str(local)


# --- BabyLM SFT Volume checkpoints (Modal ``ravens-babylm-sft``) ---

SFT_CHECKPOINTS_ROOT = "/checkpoints"


def is_sft_run_id(model_id: str) -> bool:
    """True for a bare SFT ``run_id`` (``tag__run_name__timestamp``, no ``/``)."""
    mid = model_id.strip()
    if not mid or "/" in mid or "@" in mid or ":" in mid:
        return False
    return "__" in mid


def is_sft_checkpoint_model(model_id: str) -> bool:
    """True for Volume paths under ``/checkpoints/`` or bare SFT run ids."""
    mid = model_id.strip()
    if mid.startswith(f"{SFT_CHECKPOINTS_ROOT}/"):
        return True
    return is_sft_run_id(mid)


def resolve_sft_model_id(model_id: str) -> str:
    """Map a bare SFT ``run_id`` to ``/checkpoints/<run_id>``; passthrough paths."""
    mid = model_id.strip().rstrip("/")
    if mid.startswith(f"{SFT_CHECKPOINTS_ROOT}/"):
        return mid
    if is_sft_run_id(mid):
        return f"{SFT_CHECKPOINTS_ROOT}/{mid}"
    return mid


def sft_run_id_from_model_id(model_id: str) -> str:
    """Extract ``run_id`` from a Volume path or bare id."""
    mid = resolve_sft_model_id(model_id) if is_sft_checkpoint_model(model_id) else model_id.strip()
    prefix = f"{SFT_CHECKPOINTS_ROOT}/"
    if mid.startswith(prefix):
        return mid[len(prefix) :].strip("/")
    return mid.strip("/")


def uses_completions_choice_only(model_id: str) -> bool:
    """Causal LMs using completions + structured choice for instruction choice_only."""
    return (
        is_pythia_model(model_id)
        or is_babylm_model(model_id)
        or is_childes_model(model_id)
        or is_qwen3_base_model(model_id)
        or is_olmo2_base_model(model_id)
    )


def resolve_instruction_prompt_mode(mode: str) -> str:
    """Expand ``auto`` to ``choice_only`` (default for Modal instruction eval)."""
    if mode == "auto":
        return "choice_only"
    return mode


def parse_params_billions(model_id: str) -> float | None:
    """Parse parameter count in billions from a HF model id, if recognized."""
    model_id = base_model_id(model_id)
    if is_miniberta_model(model_id):
        if "med-small" in model_id.lower():
            return _MINIBERTA_MED_SMALL_PARAMS_B
        return _MINIBERTA_BASE_PARAMS_B
    if is_childes_model(model_id) or "childes-" in model_id.lower():
        return _CHILDES_PARAMS_B
    if is_babylm_model(model_id):
        return _BABYLM_PARAMS_B
    m = _SIZE_RE.search(model_id)
    if not m:
        return None
    if m.group(3):
        return float(m.group(3))
    if m.group(4):
        return float(m.group(4))
    if m.group(5):
        return float(m.group(5))
    if m.group(6):
        return float(m.group(6))
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
    m = _CHILDES_RUNG_CORPUS_RE.search(model_id)
    if m:
        return float(m.group(1))
    short = _CHILDES_SHORT_RE.match(model_id.strip().rstrip("/"))
    if short:
        return float(short.group(1))
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


# Named ``--models`` aliases → HF id lists. Includes short synonyms (``olmo``,
# ``minibertas``) so typos do not get forwarded to vLLM as bogus repo ids.
_MODEL_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "sweep": SCALING_SWEEP_MODELS,
    "pythia": PYTHIA_SCALING_MODELS,
    "olmo": OLMO2_MODELS,
    "olmo2": OLMO2_MODELS,
    "qwen3": QWEN3_SCALING_MODELS,
    "babylm": BABYLM_SCALING_MODELS,
    "childes": CHILDES_SCALING_MODELS,
    **{
        f"childes-{b}": (childes_ladder_model_id(b),)
        for b in CHILDES_LADDER_BUDGETS
    },
    **{
        f"childes-{b.lower()}": (childes_ladder_model_id(b),)
        for b in CHILDES_LADDER_BUDGETS
    },
    "miniberta": MINIBERTA_SCALING_MODELS,
    "minibertas": MINIBERTA_SCALING_MODELS,
}


def resolve_models_arg(models: str) -> list[str]:
    """Expand ``--models`` CLI value to a list of HuggingFace model ids.

    - ``sweep`` → ``SCALING_SWEEP_MODELS``
    - ``pythia`` / ``pythia-checkpoints`` / ``olmo`` / ``olmo2`` /
      ``olmo-checkpoints`` / ``olmo2-checkpoints`` / ``qwen3`` / ``babylm`` /
      ``childes`` / ``miniberta`` → family subset
    - comma-separated HF ids → explicit list (``...@stepN`` / ``...@stage1-...``)
    - BabyLM SFT ``run_id`` (``…__…__…``) or ``/checkpoints/<run_id>`` → Volume path

    Bare names without ``/`` that are not aliases or SFT run ids raise ``ValueError``
    (avoids ``vllm serve olmo`` → ``huggingface.co/olmo`` 401 failures).
    """
    key = models.strip().lower()
    if key in _MODEL_ARG_ALIASES:
        return list(_MODEL_ARG_ALIASES[key])
    if key == "pythia-checkpoints":
        return pythia_checkpoint_model_ids()
    if key in ("olmo-checkpoints", "olmo2-checkpoints"):
        return olmo2_checkpoint_model_ids()

    raw = [m.strip() for m in models.split(",") if m.strip()]
    resolved: list[str] = []
    for mid in raw:
        if is_sft_checkpoint_model(mid):
            resolved.append(resolve_sft_model_id(mid))
        elif is_childes_model(mid):
            resolved.append(normalize_childes_model_id(mid))
        else:
            resolved.append(mid)

    alias_names = sorted(
        set(_MODEL_ARG_ALIASES)
        | {"pythia-checkpoints", "olmo-checkpoints", "olmo2-checkpoints"}
    )
    for mid in resolved:
        # HF ids contain ``/``; checkpoint forms contain ``@``; Ollama tags use ``:``.
        # SFT Volume paths also contain ``/``.
        if "/" in mid or "@" in mid or ":" in mid:
            continue
        raise ValueError(
            f"Unknown model id {mid!r}. Pass a HuggingFace id "
            f"(e.g. allenai/OLMo2-7B-1124), an SFT run_id "
            f"(e.g. babylm-10m-gpt2__all_types__20260723T212815Z), "
            f"alias babylm-sft (BabyLM Volume SFT only), "
            f"childes-sft (CHILDES Volume SFT only), "
            f"or one of the aliases: {', '.join(alias_names)}."
        )
    return resolved
