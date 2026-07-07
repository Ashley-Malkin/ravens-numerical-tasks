"""Sync per-trial eval JSON from Modal containers to local ``artifacts/runs/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ravens_numerical.paths import CONTAINER_RAVENS_ROOT, RUNS_DIR

CONTAINER_RUNS_ROOT = f"{CONTAINER_RAVENS_ROOT}/artifacts/runs"


def local_path_from_container_results(
    container_results_path: str | Path,
    *,
    local_runs_root: Path | None = None,
    container_runs_root: str = CONTAINER_RUNS_ROOT,
) -> Path:
    """Map in-container results JSON path to the matching local path."""
    local_root = local_runs_root or RUNS_DIR
    path = Path(container_results_path)
    root = Path(container_runs_root)
    try:
        rel = path.relative_to(root)
    except ValueError:
        parts = path.parts
        if "runs" in parts:
            idx = parts.index("runs")
            rel = Path(*parts[idx + 1 :])
        elif "results" in parts:
            idx = parts.index("results")
            rel = Path(*parts[idx + 1 :])
        else:
            rel = Path(path.name)
    return local_root / rel


def attach_results_data(summary: dict[str, Any], results_path: Path) -> dict[str, Any]:
    """Add ``results_data`` (trial list) for transport Modal → local machine."""
    out = dict(summary)
    out["results_data"] = json.loads(results_path.read_text(encoding="utf-8"))
    return out


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Summary safe to print/log (omit bulky trial payloads)."""
    return {k: v for k, v in summary.items() if k != "results_data"}


def save_results_local(
    summary: dict[str, Any],
    *,
    local_runs_root: Path | None = None,
    container_runs_root: str = CONTAINER_RUNS_ROOT,
) -> Path | None:
    """Write ``results_data`` to disk; set ``local_results_path`` on ``summary``."""
    data = summary.pop("results_data", None)
    if data is None:
        return None
    container_path = summary.get("results_path")
    if not container_path:
        return None
    local_path = local_path_from_container_results(
        container_path,
        local_runs_root=local_runs_root,
        container_runs_root=container_runs_root,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    summary["local_results_path"] = str(local_path)
    return local_path


def export_run_results(
    summary: dict[str, Any],
    prompt_type: str,
    *,
    local_runs_root: Path | None = None,
    container_runs_root: str = CONTAINER_RUNS_ROOT,
) -> list[Path]:
    """Save all result JSON files from one Modal run (handles ``prompt_type=both``)."""
    saved: list[Path] = []
    if prompt_type == "both":
        for pt_summary in summary.values():
            if isinstance(pt_summary, dict):
                path = save_results_local(
                    pt_summary,
                    local_runs_root=local_runs_root,
                    container_runs_root=container_runs_root,
                )
                if path is not None:
                    saved.append(path)
    elif isinstance(summary, dict):
        path = save_results_local(
            summary,
            local_runs_root=local_runs_root,
            container_runs_root=container_runs_root,
        )
        if path is not None:
            saved.append(path)
    return saved
