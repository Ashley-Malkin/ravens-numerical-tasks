"""Raven's numerical reasoning benchmark and evaluation toolkit."""

from ravens_numerical.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    LEGACY_DATA_DIR,
    REPO_ROOT,
    RUNS_DIR,
    TASKS_ABA_JSON,
    TASKS_JSON,
    TASKS_WEBB_JSON,
)

# Backward-compatible alias used by eval runner.
RESULTS_DIR = RUNS_DIR

__all__ = [
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "LEGACY_DATA_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "RUNS_DIR",
    "TASKS_ABA_JSON",
    "TASKS_JSON",
    "TASKS_WEBB_JSON",
]
