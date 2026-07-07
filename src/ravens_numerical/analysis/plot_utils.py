"""Shared helpers for experiment-log plotting scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ravens_numerical.paths import BABYLM_EXPERIMENTS_MD, REPO_ROOT


def experiment_log_paths(
    primary: Path,
    *,
    include_babylm_experiments: bool = True,
    babylm_path: Path | None = None,
) -> list[Path]:
    """Primary log; optionally also ``babyLMexperiments.md``."""
    babylm = babylm_path or BABYLM_EXPERIMENTS_MD
    paths = [primary]
    if (
        include_babylm_experiments
        and primary.resolve() != babylm.resolve()
        and babylm.is_file()
    ):
        paths.append(babylm)
    return paths


def iter_experiment_blocks(paths: list[Path]):
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        yield from re.split(r"\n---\n", text)


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
