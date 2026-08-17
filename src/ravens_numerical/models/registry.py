"""Default model IDs aligned with ``evaluate.py`` and ``baby_reasoning_eval``.

Use these with ``baby_reasoning/script/run``: ``--backend vllm`` + HF ids, or ``--backend ollama``
+ tags below.
"""

from __future__ import annotations

import re
from pathlib import Path

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

# TinyDialogues GPT-2-small finals (Google Drive → Modal Volume ``ravens-td-base``).
TD_DRIVE_PARENT_ID = "1G2jw_WbJXRTbx825oMX_5nePcBnP_ZsZ"
TD_BASE_VOLUME_NAME = "ravens-td-base"
TD_BASE_VOLUME_MOUNT = "/td-base"
TD_SEED = 42
TD_BUDGETS: tuple[str, ...] = ("10M", "20M", "50M", "100M", "200M")
TD_FOLDER_BY_BUDGET: dict[str, str] = {
    b: f"GPT2-small_TD_{b}_20-epochs_seed{TD_SEED}" for b in TD_BUDGETS
}
TD_LOGICAL_PREFIX = "tinydialogues"


def _normalize_td_budget(raw: str) -> str:
    b = raw.strip()
    if b.lower().endswith("m"):
        return b[:-1] + "M"
    return b


def td_folder_name(budget: str) -> str:
    """Drive / Volume folder name for a TD budget (e.g. ``10M``)."""
    b = _normalize_td_budget(budget)
    if b not in TD_FOLDER_BY_BUDGET:
        raise ValueError(
            f"Unknown TinyDialogues budget {budget!r}; expected one of {TD_BUDGETS}"
        )
    return TD_FOLDER_BY_BUDGET[b]


def td_model_id(budget: str) -> str:
    """Logical id: ``tinydialogues/GPT2-small_TD_{B}_20-epochs_seed42``."""
    return f"{TD_LOGICAL_PREFIX}/{td_folder_name(budget)}"


TD_SCALING_MODELS: tuple[str, ...] = tuple(td_model_id(b) for b in TD_BUDGETS)
DEFAULT_TD_VLLM = TD_SCALING_MODELS[0]

# Local download root (repo-relative); Modal containers use ``TD_BASE_VOLUME_MOUNT``.
TD_LOCAL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "tinydialogues"

# OpenSubtitles GPT-2-small 20M (local ``open-subtitles-models/`` → Volume ``ravens-os-base``).
OS_BASE_VOLUME_NAME = "ravens-os-base"
OS_BASE_VOLUME_MOUNT = "/os-base"
OS_BUDGET = "20M"
OS_SEEDS: tuple[int, ...] = (0, 42, 123)
OS_FOLDER_BY_SEED: dict[int, str] = {
    s: f"GPT2-small_opensubtitles_{OS_BUDGET}_1e-04_20-epochs_seed{s}" for s in OS_SEEDS
}
OS_LOGICAL_PREFIX = "opensubtitles"
OS_LOCAL_ROOT = Path(__file__).resolve().parents[3] / "open-subtitles-models"

# Wiki GPT-2-small 20M (local ``wiki-models/`` → Volume ``ravens-wiki-base``).
WIKI_BASE_VOLUME_NAME = "ravens-wiki-base"
WIKI_BASE_VOLUME_MOUNT = "/wiki-base"
WIKI_BUDGET = "20M"
WIKI_SEEDS: tuple[int, ...] = (0, 42, 123)
WIKI_FOLDER_BY_SEED: dict[int, str] = {
    s: f"GPT2-small_wiki_{WIKI_BUDGET}_1e-04_20-epochs_seed{s}" for s in WIKI_SEEDS
}
WIKI_LOGICAL_PREFIX = "wiki"
WIKI_LOCAL_ROOT = Path(__file__).resolve().parents[3] / "wiki-models"

# TinyStories GPT-2-small 10M (local ``tiny-stories-model/`` → Volume ``ravens-ts-base``).
TS_BASE_VOLUME_NAME = "ravens-ts-base"
TS_BASE_VOLUME_MOUNT = "/ts-base"
TS_BUDGET = "10M"
TS_FOLDER_NAME = "GPT2-small_tinystories_10m_1e-04"
TS_LOGICAL_PREFIX = "tinystories"
TS_LOCAL_ROOT = Path(__file__).resolve().parents[3] / "tiny-stories-model"


def os_folder_name(seed: int | str) -> str:
    s = int(seed)
    if s not in OS_FOLDER_BY_SEED:
        raise ValueError(f"Unknown OpenSubtitles seed {seed!r}; expected one of {OS_SEEDS}")
    return OS_FOLDER_BY_SEED[s]


def os_model_id(seed: int | str) -> str:
    """Logical id: ``opensubtitles/GPT2-small_opensubtitles_20M_…_seed{S}``."""
    return f"{OS_LOGICAL_PREFIX}/{os_folder_name(seed)}"


def wiki_folder_name(seed: int | str) -> str:
    s = int(seed)
    if s not in WIKI_FOLDER_BY_SEED:
        raise ValueError(f"Unknown Wiki seed {seed!r}; expected one of {WIKI_SEEDS}")
    return WIKI_FOLDER_BY_SEED[s]


def wiki_model_id(seed: int | str) -> str:
    """Logical id: ``wiki/GPT2-small_wiki_20M_…_seed{S}``."""
    return f"{WIKI_LOGICAL_PREFIX}/{wiki_folder_name(seed)}"


def ts_folder_name() -> str:
    return TS_FOLDER_NAME


def ts_model_id() -> str:
    """Logical id: ``tinystories/GPT2-small_tinystories_10m_1e-04``."""
    return f"{TS_LOGICAL_PREFIX}/{TS_FOLDER_NAME}"


OS_SCALING_MODELS: tuple[str, ...] = tuple(os_model_id(s) for s in OS_SEEDS)
DEFAULT_OS_VLLM = OS_SCALING_MODELS[1] if 42 in OS_SEEDS else OS_SCALING_MODELS[0]
WIKI_SCALING_MODELS: tuple[str, ...] = tuple(wiki_model_id(s) for s in WIKI_SEEDS)
DEFAULT_WIKI_VLLM = WIKI_SCALING_MODELS[1] if 42 in WIKI_SEEDS else WIKI_SCALING_MODELS[0]
TS_SCALING_MODELS: tuple[str, ...] = (ts_model_id(),)
DEFAULT_TS_VLLM = TS_SCALING_MODELS[0]

_MINIBERTA_CORPUS_SIZES = ("1M", "10M", "100M", "1B")
_MINIBERTA_SEEDS = (1, 2, 3)

MINIBERTA_SCALING_MODELS: tuple[str, ...] = tuple(
    f"nyu-mll/roberta-{'med-small-' if size == '1M' else 'base-'}{size}-{seed}"
    for size in _MINIBERTA_CORPUS_SIZES
    for seed in _MINIBERTA_SEEDS
)

DEFAULT_MINIBERTA_VLLM = MINIBERTA_SCALING_MODELS[0]

# MiniBERTa MLM SFT ladder: all corpus sizes at seed 1.
MINIBERTA_SFT_SEED = 1
MINIBERTA_SFT_SIZES: tuple[str, ...] = _MINIBERTA_CORPUS_SIZES


def normalize_miniberta_size(raw: str) -> str:
    """Normalize ``1m`` / ``1M`` / ``1b`` → ``1M`` / ``1B``."""
    s = raw.strip()
    if len(s) < 2:
        raise ValueError(f"Invalid MiniBERTa size {raw!r}")
    unit = s[-1].upper()
    if unit not in ("M", "B"):
        raise ValueError(f"Invalid MiniBERTa size {raw!r}; expected …M or …B")
    return s[:-1] + unit


def miniberta_hub_id(size: str, seed: int = MINIBERTA_SFT_SEED) -> str:
    """Hub id for a MiniBERTa size/seed (``nyu-mll/roberta-…``)."""
    size_n = normalize_miniberta_size(size)
    if size_n not in MINIBERTA_SFT_SIZES:
        raise ValueError(
            f"Unknown MiniBERTa size {size!r}; expected one of {MINIBERTA_SFT_SIZES}"
        )
    arch = "med-small-" if size_n == "1M" else "base-"
    return f"nyu-mll/roberta-{arch}{size_n}-{int(seed)}"


MINIBERTA_SFT_MODELS: tuple[str, ...] = tuple(
    miniberta_hub_id(s, MINIBERTA_SFT_SEED) for s in MINIBERTA_SFT_SIZES
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
    QWEN3_8B_BASE: "A10G",
    "Qwen/Qwen3-14B": "A100",
    "BabyLM-community/babylm-baseline-10m-gpt2": "T4",
    "BabyLM-community/babylm-baseline-100m-gpt2": "T4",
    **{mid: "T4" for mid in CHILDES_SCALING_MODELS},
    **{mid: "T4" for mid in TD_SCALING_MODELS},
    **{mid: "T4" for mid in OS_SCALING_MODELS},
    **{mid: "T4" for mid in WIKI_SCALING_MODELS},
    **{mid: "T4" for mid in TS_SCALING_MODELS},
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

# TinyDialogues: GPT-2-small (~124M params), context 1024.
_TD_PARAMS_B = 0.124
_TD_MAX_LEN = 1024
_TD_SHORT_RE = re.compile(r"^td-(\d+(?:\.\d+)?)[mM]$", re.I)
_TD_FOLDER_RE = re.compile(
    r"^GPT2-small_TD_(\d+(?:\.\d+)?)[mM]_20-epochs_seed(\d+)$",
    re.I,
)
_TD_CORPUS_RE = re.compile(r"TD_(\d+(?:\.\d+)?)[mM]_", re.I)

# OpenSubtitles / Wiki: GPT-2-small (~124M), context 1024, seed ladder at 20M.
_OS_WIKI_PARAMS_B = 0.124
_OS_WIKI_MAX_LEN = 1024
_OS_SHORT_RE = re.compile(r"^(?:os|opensubtitles)(?:-seed)?-?(\d+)$", re.I)
_OS_FOLDER_RE = re.compile(
    r"^GPT2-small_opensubtitles_20M_1e-04_20-epochs_seed(\d+)$",
    re.I,
)
_WIKI_SHORT_RE = re.compile(r"^wiki(?:-seed)?-?(\d+)$", re.I)
_WIKI_FOLDER_RE = re.compile(
    r"^GPT2-small_wiki_20M_1e-04_20-epochs_seed(\d+)$",
    re.I,
)

# TinyStories: GPT-2-small (~124M), context 1024, single 10M checkpoint.
_TS_PARAMS_B = 0.124
_TS_MAX_LEN = 1024
_TS_SHORT_RE = re.compile(r"^(?:ts|tinystories)(?:-10[mM])?$", re.I)
_TS_FOLDER_RE = re.compile(r"^GPT2-small_tinystories_10m_1e-04$", re.I)

_MINIBERTA_MED_SMALL_PARAMS_B = 0.045
_MINIBERTA_BASE_PARAMS_B = 0.125
_MINIBERTA_MAX_LEN = 512

_MINIBERTA_CORPUS_RE = re.compile(
    r"roberta-(?:med-small-|base-)(\d+)([mMbB])-", re.I
)
_MINIBERTA_SEED_RE = re.compile(r"-(\d+)$")
_MINIBERTA_SHORT_RE = re.compile(r"^miniberta-(\d+)([mMbB])$", re.I)
_MINIBERTA_TAG_RE = re.compile(
    r"^miniberta-(\d+)([mMbB])-seed(\d+)(?:__|$)",
    re.I,
)


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
    if (
        is_childes_model(model_id)
        or is_td_model(model_id)
        or is_os_model(model_id)
        or is_wiki_model(model_id)
        or is_ts_model(model_id)
    ):
        return "T4"
    return MODEL_GPU_TIER.get(base_model_id(model_id), DEFAULT_GPU_TIER)


def max_model_len_for_model(model_id: str, default: int = 2048) -> int:
    """vLLM context length per model (respects architecture limits)."""
    model_id = base_model_id(model_id)
    if is_miniberta_model(model_id):
        return _MINIBERTA_MAX_LEN
    # MiniBERTa SFT Volume paths / run_ids (``miniberta-10M-seed1__…``).
    if "miniberta-" in model_id.lower() or "nyu-mll/roberta-" in model_id.lower():
        return _MINIBERTA_MAX_LEN
    # Base ladder ids + SFT run_ids / Volume paths (``.../childes-seed42-rung…``).
    if is_childes_model(model_id) or "childes-" in model_id.lower():
        return _CHILDES_MAX_LEN
    if is_td_model(model_id) or "td-" in model_id.lower() or "tinydialogues" in model_id.lower():
        return _TD_MAX_LEN
    if (
        is_os_model(model_id)
        or "os-seed" in model_id.lower()
        or "opensubtitles" in model_id.lower()
    ):
        return _OS_WIKI_MAX_LEN
    # Base wiki ids + SFT Volume paths (``/checkpoints/wiki-seed42-20M__…``).
    if (
        is_wiki_model(model_id)
        or "wiki-" in model_id.lower()
        or "/wiki/" in model_id.lower()
    ):
        return _OS_WIKI_MAX_LEN
    # Base TinyStories + SFT Volume paths (``/checkpoints/ts-10M__…``).
    if (
        is_ts_model(model_id)
        or "tinystories" in model_id.lower()
        or re.search(r"(?:^|/)ts-10[mM]", model_id, re.I)
        or model_id.lower().startswith("ts-10")
        or "/ts-10" in model_id.lower()
    ):
        return _TS_MAX_LEN
    if "babylm" in model_id.lower():
        return _BABYLM_MAX_LEN
    params_b = parse_params_billions(model_id)
    if params_b is not None and params_b >= _LARGE_MODEL_PARAMS_B:
        return _LARGE_MODEL_MAX_LEN
    return default


def model_family(model_id: str) -> str:
    model_id = base_model_id(model_id)
    if is_miniberta_model(model_id) or "miniberta-" in model_id.lower():
        return "miniberta"
    if is_childes_model(model_id) or "childes-" in model_id.lower():
        return "childes"
    if is_td_model(model_id) or model_id.lower().startswith("td-") or "tinydialogues" in model_id.lower():
        return "td"
    if (
        is_os_model(model_id)
        or "os-seed" in model_id.lower()
        or "opensubtitles" in model_id.lower()
    ):
        return "os"
    if (
        is_wiki_model(model_id)
        or "wiki-" in model_id.lower()
        or model_id.lower().startswith("wiki/")
    ):
        return "wiki"
    if (
        is_ts_model(model_id)
        or "tinystories" in model_id.lower()
        or re.search(r"(?:^|/)ts-10[mM]", model_id, re.I)
        or model_id.lower().startswith("ts-10")
        or "/ts-10" in model_id.lower()
    ):
        return "ts"
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
    mid = model_id.lower().strip().rstrip("/")
    # Short aliases like miniberta-10M (not SFT run_ids with ``__``).
    if _MINIBERTA_SHORT_RE.match(mid):
        return True
    return mid.startswith("nyu-mll/roberta-") and (
        "med-small-" in mid or re.search(r"roberta-base-\d+[mMbB]-", mid) is not None
    )


def is_miniberta_sft_checkpoint(model_id: str) -> bool:
    """True for Volume MiniBERTa SFT run_ids / ``/checkpoints/miniberta-…`` paths."""
    if not is_sft_checkpoint_model(model_id):
        return False
    return sft_run_id_from_model_id(model_id).lower().startswith("miniberta-")


def normalize_miniberta_model_id(model_id: str) -> str:
    """Expand ``miniberta-10M`` → Hub id at ``MINIBERTA_SFT_SEED``; passthrough Hub ids."""
    mid = model_id.strip().rstrip("/")
    if "__" in mid and "/" not in mid:
        return mid
    short = _MINIBERTA_SHORT_RE.match(mid)
    if short:
        size = normalize_miniberta_size(short.group(1) + short.group(2))
        return miniberta_hub_id(size, MINIBERTA_SFT_SEED)
    return mid


def miniberta_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``miniberta-10M-seed1``, or ``None``."""
    mid = model_id.strip().rstrip("/")
    tag_m = _MINIBERTA_TAG_RE.match(mid)
    if tag_m:
        size = normalize_miniberta_size(tag_m.group(1) + tag_m.group(2))
        return f"miniberta-{size}-seed{tag_m.group(3)}"
    short = _MINIBERTA_SHORT_RE.match(mid)
    if short:
        size = normalize_miniberta_size(short.group(1) + short.group(2))
        return f"miniberta-{size}-seed{MINIBERTA_SFT_SEED}"
    hub = _MINIBERTA_CORPUS_RE.search(mid)
    if hub and mid.lower().startswith("nyu-mll/roberta-"):
        size = normalize_miniberta_size(hub.group(1) + hub.group(2))
        seed = parse_miniberta_seed(mid)
        if seed is None:
            seed = MINIBERTA_SFT_SEED
        return f"miniberta-{size}-seed{seed}"
    return None


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


def parse_td_model_id(model_id: str) -> str | None:
    """Parse a TinyDialogues id into the Drive/Volume folder name, or ``None``.

    Accepts short aliases (``td-10M``), logical ids
    (``tinydialogues/GPT2-small_TD_10M_20-epochs_seed42``), bare folder names,
    and Volume paths under ``/td-base/``.
    """
    mid = model_id.strip().rstrip("/")
    if mid.startswith(f"{TD_BASE_VOLUME_MOUNT}/"):
        mid = mid[len(TD_BASE_VOLUME_MOUNT) :].lstrip("/")
    short = _TD_SHORT_RE.match(mid)
    if short:
        budget = _normalize_td_budget(short.group(1) + "M")
        if budget in TD_FOLDER_BY_BUDGET:
            return TD_FOLDER_BY_BUDGET[budget]
        return None
    prefix = f"{TD_LOGICAL_PREFIX}/"
    if mid.startswith(prefix):
        mid = mid[len(prefix) :]
    folder_match = _TD_FOLDER_RE.match(mid)
    if folder_match:
        budget = _normalize_td_budget(folder_match.group(1) + "M")
        if budget in TD_FOLDER_BY_BUDGET:
            return TD_FOLDER_BY_BUDGET[budget]
    if mid in TD_FOLDER_BY_BUDGET.values():
        return mid
    return None


def normalize_td_model_id(model_id: str) -> str:
    """Expand short TD aliases to the full logical ``tinydialogues/...`` id."""
    folder = parse_td_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    return f"{TD_LOGICAL_PREFIX}/{folder}"


def is_td_model(model_id: str) -> bool:
    """True for TinyDialogues base ids (not SFT run_ids)."""
    return parse_td_model_id(model_id) is not None


def td_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``td-seed42-10M``, or ``None`` if not TD."""
    folder = parse_td_model_id(model_id)
    if folder is None:
        return None
    m = _TD_FOLDER_RE.match(folder)
    if not m:
        return None
    budget = _normalize_td_budget(m.group(1) + "M")
    seed = m.group(2)
    return f"td-seed{seed}-{budget}"


def assert_td_checkpoint_dir(path: Path | str) -> Path:
    """Require HF-loadable files in a checkpoint directory."""
    return assert_hf_checkpoint_dir(path, label="checkpoint")


def assert_hf_checkpoint_dir(path: Path | str, *, label: str = "checkpoint") -> Path:
    """Require HF-loadable files (config + weights + tokenizer) in a directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {root}")
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Missing config.json in {label}: {root}")
    has_weights = any(
        root.glob(pattern)
        for pattern in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        has_weights = bool(list(root.glob("*.safetensors")) or list(root.glob("pytorch_model*.bin")))
    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found in {label}: {root} "
            "(expected model.safetensors / pytorch_model.bin)"
        )
    has_tok = any(
        (root / name).exists()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
        )
    )
    if not has_tok:
        raise FileNotFoundError(
            f"No tokenizer files found in {label}: {root}."
        )
    return root


def resolve_td_local_path(model_id: str) -> str:
    """Resolve a TD id to a filesystem path (Volume mount or local download).

    Preference order:
    1. ``/td-base/<folder>`` when present (Modal Volume)
    2. ``artifacts/models/tinydialogues/<folder>`` on the local/repo tree
    """
    folder = parse_td_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")

    volume_path = Path(TD_BASE_VOLUME_MOUNT) / folder
    if volume_path.is_dir():
        return str(assert_hf_checkpoint_dir(volume_path, label="TD checkpoint"))

    local_path = TD_LOCAL_ROOT / folder
    if local_path.is_dir():
        return str(assert_hf_checkpoint_dir(local_path, label="TD checkpoint"))

    raise FileNotFoundError(
        f"TinyDialogues checkpoint not found for {model_id!r}. "
        f"Expected {volume_path} (Volume {TD_BASE_VOLUME_NAME!r}) or {local_path}. "
        "Run: modal run babylm_finetune/scripts/upload_td_base_volume.py"
    )


def parse_os_model_id(model_id: str) -> str | None:
    """Parse an OpenSubtitles id into the Volume/local folder name, or ``None``."""
    mid = model_id.strip().rstrip("/")
    if mid.startswith(f"{OS_BASE_VOLUME_MOUNT}/"):
        mid = mid[len(OS_BASE_VOLUME_MOUNT) :].lstrip("/")
    # Avoid matching SFT run_ids like os-seed42-20M__all_types__…
    if "__" in mid and "/" not in mid:
        return None
    short = _OS_SHORT_RE.match(mid)
    if short:
        seed = int(short.group(1))
        if seed in OS_FOLDER_BY_SEED:
            return OS_FOLDER_BY_SEED[seed]
        return None
    prefix = f"{OS_LOGICAL_PREFIX}/"
    if mid.startswith(prefix):
        mid = mid[len(prefix) :]
    folder_match = _OS_FOLDER_RE.match(mid)
    if folder_match:
        seed = int(folder_match.group(1))
        if seed in OS_FOLDER_BY_SEED:
            return OS_FOLDER_BY_SEED[seed]
    if mid in OS_FOLDER_BY_SEED.values():
        return mid
    return None


def normalize_os_model_id(model_id: str) -> str:
    folder = parse_os_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    return f"{OS_LOGICAL_PREFIX}/{folder}"


def is_os_model(model_id: str) -> bool:
    return parse_os_model_id(model_id) is not None


def os_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``os-seed42-20M``, or ``None``."""
    folder = parse_os_model_id(model_id)
    if folder is None:
        return None
    m = _OS_FOLDER_RE.match(folder)
    if not m:
        return None
    return f"os-seed{m.group(1)}-{OS_BUDGET}"


def resolve_os_local_path(model_id: str) -> str:
    folder = parse_os_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    volume_path = Path(OS_BASE_VOLUME_MOUNT) / folder
    if volume_path.is_dir():
        return str(assert_hf_checkpoint_dir(volume_path, label="OpenSubtitles checkpoint"))
    local_path = OS_LOCAL_ROOT / folder
    if local_path.is_dir():
        return str(assert_hf_checkpoint_dir(local_path, label="OpenSubtitles checkpoint"))
    raise FileNotFoundError(
        f"OpenSubtitles checkpoint not found for {model_id!r}. "
        f"Expected {volume_path} (Volume {OS_BASE_VOLUME_NAME!r}) or {local_path}. "
        "Run: modal run babylm_finetune/scripts/upload_opensubtitles_base_volume.py"
    )


def parse_wiki_model_id(model_id: str) -> str | None:
    """Parse a Wiki id into the Volume/local folder name, or ``None``."""
    mid = model_id.strip().rstrip("/")
    if mid.startswith(f"{WIKI_BASE_VOLUME_MOUNT}/"):
        mid = mid[len(WIKI_BASE_VOLUME_MOUNT) :].lstrip("/")
    if "__" in mid and "/" not in mid:
        return None
    short = _WIKI_SHORT_RE.match(mid)
    if short:
        seed = int(short.group(1))
        if seed in WIKI_FOLDER_BY_SEED:
            return WIKI_FOLDER_BY_SEED[seed]
        return None
    prefix = f"{WIKI_LOGICAL_PREFIX}/"
    if mid.startswith(prefix):
        mid = mid[len(prefix) :]
    folder_match = _WIKI_FOLDER_RE.match(mid)
    if folder_match:
        seed = int(folder_match.group(1))
        if seed in WIKI_FOLDER_BY_SEED:
            return WIKI_FOLDER_BY_SEED[seed]
    if mid in WIKI_FOLDER_BY_SEED.values():
        return mid
    return None


def normalize_wiki_model_id(model_id: str) -> str:
    folder = parse_wiki_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    return f"{WIKI_LOGICAL_PREFIX}/{folder}"


def is_wiki_model(model_id: str) -> bool:
    return parse_wiki_model_id(model_id) is not None


def wiki_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``wiki-seed42-20M``, or ``None``."""
    folder = parse_wiki_model_id(model_id)
    if folder is None:
        return None
    m = _WIKI_FOLDER_RE.match(folder)
    if not m:
        return None
    return f"wiki-seed{m.group(1)}-{WIKI_BUDGET}"


def resolve_wiki_local_path(model_id: str) -> str:
    folder = parse_wiki_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    volume_path = Path(WIKI_BASE_VOLUME_MOUNT) / folder
    if volume_path.is_dir():
        return str(assert_hf_checkpoint_dir(volume_path, label="Wiki checkpoint"))
    local_path = WIKI_LOCAL_ROOT / folder
    if local_path.is_dir():
        return str(assert_hf_checkpoint_dir(local_path, label="Wiki checkpoint"))
    raise FileNotFoundError(
        f"Wiki checkpoint not found for {model_id!r}. "
        f"Expected {volume_path} (Volume {WIKI_BASE_VOLUME_NAME!r}) or {local_path}. "
        "Run: modal run babylm_finetune/scripts/upload_wiki_base_volume.py"
    )


def parse_ts_model_id(model_id: str) -> str | None:
    """Parse a TinyStories id into the Volume/local folder name, or ``None``."""
    mid = model_id.strip().rstrip("/")
    if mid.startswith(f"{TS_BASE_VOLUME_MOUNT}/"):
        mid = mid[len(TS_BASE_VOLUME_MOUNT) :].lstrip("/")
    # Avoid matching SFT run_ids like ts-10M__all_types__…
    if "__" in mid and "/" not in mid:
        return None
    if _TS_SHORT_RE.match(mid):
        return TS_FOLDER_NAME
    prefix = f"{TS_LOGICAL_PREFIX}/"
    if mid.startswith(prefix):
        mid = mid[len(prefix) :]
    if _TS_FOLDER_RE.match(mid) or mid == TS_FOLDER_NAME:
        return TS_FOLDER_NAME
    return None


def normalize_ts_model_id(model_id: str) -> str:
    folder = parse_ts_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    return f"{TS_LOGICAL_PREFIX}/{folder}"


def is_ts_model(model_id: str) -> bool:
    return parse_ts_model_id(model_id) is not None


def ts_model_tag(model_id: str) -> str | None:
    """Short run_id tag like ``ts-10M``, or ``None``."""
    if parse_ts_model_id(model_id) is None:
        return None
    return f"ts-{TS_BUDGET}"


def resolve_ts_local_path(model_id: str) -> str:
    folder = parse_ts_model_id(model_id)
    if folder is None:
        return model_id.strip().rstrip("/")
    volume_path = Path(TS_BASE_VOLUME_MOUNT) / folder
    if volume_path.is_dir():
        return str(assert_hf_checkpoint_dir(volume_path, label="TinyStories checkpoint"))
    local_path = TS_LOCAL_ROOT / folder
    if local_path.is_dir():
        return str(assert_hf_checkpoint_dir(local_path, label="TinyStories checkpoint"))
    raise FileNotFoundError(
        f"TinyStories checkpoint not found for {model_id!r}. "
        f"Expected {volume_path} (Volume {TS_BASE_VOLUME_NAME!r}) or {local_path}. "
        "Run: modal run babylm_finetune/scripts/upload_tinystories_base_volume.py"
    )


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
        or is_td_model(model_id)
        or is_os_model(model_id)
        or is_wiki_model(model_id)
        or is_ts_model(model_id)
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
    if is_td_model(model_id) or "td-" in model_id.lower() or "tinydialogues" in model_id.lower():
        return _TD_PARAMS_B
    if (
        is_os_model(model_id)
        or "os-seed" in model_id.lower()
        or "opensubtitles" in model_id.lower()
        or is_wiki_model(model_id)
        or "wiki-" in model_id.lower()
    ):
        return _OS_WIKI_PARAMS_B
    if (
        is_ts_model(model_id)
        or "tinystories" in model_id.lower()
        or re.search(r"(?:^|/)ts-10[mM]", model_id, re.I)
        or model_id.lower().startswith("ts-10")
        or "/ts-10" in model_id.lower()
    ):
        return _TS_PARAMS_B
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
    m = _TD_CORPUS_RE.search(model_id)
    if m:
        return float(m.group(1))
    short_td = _TD_SHORT_RE.match(model_id.strip().rstrip("/"))
    if short_td:
        return float(short_td.group(1))
    # SFT tags like td-seed42-10M__all_types__…
    m = re.search(r"td-seed\d+-(\d+(?:\.\d+)?)[mM](?:__|$|[^A-Za-z0-9])", model_id, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"td-seed\d+-(\d+(?:\.\d+)?)[mM]$", model_id, re.I)
    if m:
        return float(m.group(1))
    # OpenSubtitles / Wiki are fixed 20M corpus.
    if (
        is_os_model(model_id)
        or "os-seed" in model_id.lower()
        or "opensubtitles" in model_id.lower()
        or is_wiki_model(model_id)
        or model_id.lower().startswith("wiki-")
        or re.search(r"wiki-seed\d+", model_id, re.I)
    ):
        return 20.0
    if (
        is_ts_model(model_id)
        or "tinystories" in model_id.lower()
        or re.search(r"(?:^|/)ts-10[mM]", model_id, re.I)
        or model_id.lower().startswith("ts-10")
        or "/ts-10" in model_id.lower()
    ):
        return 10.0
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
    "td": TD_SCALING_MODELS,
    "tinydialogues": TD_SCALING_MODELS,
    **{f"td-{b}": (td_model_id(b),) for b in TD_BUDGETS},
    **{f"td-{b.lower()}": (td_model_id(b),) for b in TD_BUDGETS},
    "os": OS_SCALING_MODELS,
    "opensubtitles": OS_SCALING_MODELS,
    **{f"os-seed{s}": (os_model_id(s),) for s in OS_SEEDS},
    **{f"os-{s}": (os_model_id(s),) for s in OS_SEEDS},
    **{f"opensubtitles-seed{s}": (os_model_id(s),) for s in OS_SEEDS},
    "wiki": WIKI_SCALING_MODELS,
    **{f"wiki-seed{s}": (wiki_model_id(s),) for s in WIKI_SEEDS},
    **{f"wiki-{s}": (wiki_model_id(s),) for s in WIKI_SEEDS},
    "ts": TS_SCALING_MODELS,
    "tinystories": TS_SCALING_MODELS,
    "ts-10m": TS_SCALING_MODELS,
    "ts-10M": TS_SCALING_MODELS,
    "miniberta": MINIBERTA_SCALING_MODELS,
    "minibertas": MINIBERTA_SCALING_MODELS,
    **{f"miniberta-{s}": (miniberta_hub_id(s),) for s in MINIBERTA_SFT_SIZES},
    **{f"miniberta-{s.lower()}": (miniberta_hub_id(s),) for s in MINIBERTA_SFT_SIZES},
}


def resolve_models_arg(models: str) -> list[str]:
    """Expand ``--models`` CLI value to a list of HuggingFace model ids.

    - ``sweep`` → ``SCALING_SWEEP_MODELS``
    - ``pythia`` / ``pythia-checkpoints`` / ``olmo`` / ``olmo2`` /
      ``olmo-checkpoints`` / ``olmo2-checkpoints`` / ``qwen3`` / ``babylm`` /
      ``childes`` / ``td`` / ``os`` / ``wiki`` / ``ts`` / ``tinystories`` /
      ``miniberta`` → family subset
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
        elif is_td_model(mid):
            resolved.append(normalize_td_model_id(mid))
        elif is_os_model(mid):
            resolved.append(normalize_os_model_id(mid))
        elif is_wiki_model(mid):
            resolved.append(normalize_wiki_model_id(mid))
        elif is_ts_model(mid):
            resolved.append(normalize_ts_model_id(mid))
        elif is_miniberta_model(mid):
            resolved.append(normalize_miniberta_model_id(mid))
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
            f"babylm-sft-alltasks (BabyLM ``__all_types__`` SFT only), "
            f"childes-sft (CHILDES Volume SFT only), "
            f"td-sft (TinyDialogues Volume SFT only), "
            f"os-sft / wiki-sft (OpenSubtitles / Wiki Volume SFT only), "
            f"ts-sft / tinystories-sft (TinyStories Volume SFT only), "
            f"miniberta-sft (MiniBERTa Volume MLM SFT only), "
            f"or one of the aliases: {', '.join(alias_names)}."
        )
    return resolved
