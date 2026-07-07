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
RUNS_DIR = ARTIFACTS_DIR / "runs"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
AGGREGATE_DIR = ARTIFACTS_DIR / "aggregate"
EXPERIMENTS_MD = LOGS_DIR / "experiments.md"
BABYLM_EXPERIMENTS_MD = LOGS_DIR / "babyLMexperiments.md"
SCALING_SUMMARY_MD = REPO_ROOT / "docs" / "scaling_summary.md"
SCALING_RESULTS_CSV = AGGREGATE_DIR / "scaling_results.csv"

# Modal container layout (repo copied to /root/ravens).
CONTAINER_RAVENS_ROOT = Path("/root/ravens")
