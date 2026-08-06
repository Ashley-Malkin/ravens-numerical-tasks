"""Repository and artifact path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    override = os.environ.get("RAVENS_ARTIFACTS_DIR")
    if override:
        return Path(override)
    return repo_root() / "artifacts"


REPO_ROOT = repo_root()
ARTIFACTS_DIR = artifacts_dir()
DATA_DIR = REPO_ROOT / "data"
LEGACY_DATA_DIR = DATA_DIR / "legacy"
TASKS_JSON = DATA_DIR / "tasks.json"
TASKS_ABA_JSON = DATA_DIR / "tasks_aba.json"
TASKS_WEBB_JSON = DATA_DIR / "tasks_webb.json"
RUNS_DIR = ARTIFACTS_DIR / "runs"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
AGGREGATE_DIR = ARTIFACTS_DIR / "aggregate"
EXPERIMENTS_MD = LOGS_DIR / "experiments.md"
BABYLM_EXPERIMENTS_MD = LOGS_DIR / "babyLMexperiments.md"
CHILDES_EXPERIMENTS_MD = LOGS_DIR / "childesExperiments.md"
PYTHIA_CHECKPOINTS_MD = LOGS_DIR / "pythiacheckpoints.md"
OLMO_CHECKPOINTS_MD = LOGS_DIR / "olmocheckpoints.md"
SCALING_SUMMARY_MD = REPO_ROOT / "docs" / "scaling_summary.md"
SCALING_RESULTS_CSV = AGGREGATE_DIR / "scaling_results.csv"

BABYLM_FINETUNE_DIR = REPO_ROOT / "babylm_finetune"
BABYLM_FINETUNE_LOGS_DIR = BABYLM_FINETUNE_DIR / "logs"
BABYLM_SFT_ALL_EVALS_MD = BABYLM_FINETUNE_LOGS_DIR / "all_sft_evals.md"
CHILDES_SFT_ALL_EVALS_MD = BABYLM_FINETUNE_LOGS_DIR / "childes_sft_evals.md"
BABYLM_SFT_N_SCALING_EVALS_MD = BABYLM_FINETUNE_LOGS_DIR / "n_scaling_evals.md"
BABYLM_FINETUNE_OUTPUTS_DIR = BABYLM_FINETUNE_DIR / "outputs"
BABYLM_SFT_N_SCALING_MANIFEST = BABYLM_FINETUNE_OUTPUTS_DIR / "n_scaling_manifest.json"
BABYLM_SFT_N_SCALING_RESULTS = BABYLM_FINETUNE_OUTPUTS_DIR / "n_scaling_results.json"


def babylm_sft_experiments_md(model_id: str, n_examples: int | None = None) -> Path:
    """Per-run_id markdown log under ``babylm_finetune/logs/<run_id>__n{N}.md``.

    When ``n_examples`` is omitted, returns the legacy ``<run_id>.md`` path
    (for callers that only need the run_id stem). Eval logging always passes
    ``n_examples`` so different ICL shot counts write distinct files.
    """
    from ravens_numerical.models.registry import sft_run_id_from_model_id

    run_id = sft_run_id_from_model_id(model_id)
    safe = run_id.replace("/", "_").replace(":", "_")
    if n_examples is None:
        return BABYLM_FINETUNE_LOGS_DIR / f"{safe}.md"
    return BABYLM_FINETUNE_LOGS_DIR / f"{safe}__n{n_examples}.md"


def combined_sft_log_path_for_n_examples(
    path: Path | None,
    n_examples: int,
    *,
    default: Path | None = None,
) -> Path:
    """Resolve a combined SFT log path that encodes ``n_examples`` in the name.

    If ``path`` (or ``default``) already ends with ``_n<digits>.md``, it is kept.
    Otherwise ``_n{n_examples}`` is inserted before the suffix so n=0 / n=1 / n=3
    do not overwrite each other.
    """
    import re

    base = path if path is not None else (default or BABYLM_SFT_ALL_EVALS_MD)
    if re.search(r"_n\d+\.md$", base.name):
        return base
    return base.with_name(f"{base.stem}_n{n_examples}{base.suffix}")


# Modal container layout (repo copied to /root/ravens).
CONTAINER_RAVENS_ROOT = Path("/root/ravens")
