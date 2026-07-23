#!/usr/bin/env python3
"""Seven-family parameter scaling plot for Easy Ravens, Webb, and ABA.

Uses the most recent ``max_tasks=140`` forced-choice result for each task/model.
Checkpoint facets show every checkpoint faintly and connect the latest training
checkpoint for each parameter size.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from ravens_numerical.models.registry import (
    base_model_id,
    checkpoint_step_from_model_id,
    is_babylm_model,
    is_miniberta_model,
    is_olmo2_checkpoint_model_id,
    is_olmo2_model,
    is_pythia_checkpoint_model_id,
    is_pythia_model,
    is_qwen3_model,
    olmo2_tokens_billions_from_revision,
    parse_checkpoint_model_id,
    parse_params_billions,
)
from ravens_numerical.paths import (
    BABYLM_EXPERIMENTS_MD,
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PLOTS_DIR,
    PYTHIA_CHECKPOINTS_MD,
)

_TASKS = ("webb", "ravens", "aba")
_TASK_LABELS = {"webb": "Webb", "ravens": "Easy Ravens", "aba": "ABA"}
_TASK_COLORS = {"webb": "#1976d2", "ravens": "#2e8b57", "aba": "#7b2cbf"}
_FACETS = (
    "pythia",
    "pythia_checkpoints",
    "olmo",
    "olmo_checkpoints",
    "qwen3",
    "miniberta",
    "babylm",
)
_FACET_LABELS = {
    "pythia": "Pythia",
    "pythia_checkpoints": "Pythia checkpoints",
    "olmo": "OLMo 2",
    "olmo_checkpoints": "OLMo 2 checkpoints",
    "qwen3": "Qwen3",
    "miniberta": "MiniBERTa",
    "babylm": "BabyLM",
}
_MAX_TASKS = "(max_tasks=140)"
_EXCLUDED_OLMO_TOKEN_BUDGETS_B = {0.0, 51.0}

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")


@dataclass(frozen=True)
class Result:
    task: str
    model_id: str
    params_b: float
    accuracy: float


def _iter_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    for part in re.split(
        r"(?=^## \d{4}-\d{2}-\d{2} )", text, flags=re.MULTILINE
    ):
        part = part.strip()
        if part.startswith("## "):
            yield part


def _header(block: str) -> str:
    return next((line for line in block.splitlines() if line.strip()), "")


def _task(block: str) -> str | None:
    for task in _TASKS:
        if f"task-type {task}" in block or f", {task})" in _header(block):
            return task
    return None


def parse_latest(*paths: Path) -> list[Result]:
    """Parse the chronologically latest matching block per task/model."""
    latest: dict[tuple[str, str], Result] = {}
    for path in paths:
        if not path.is_file():
            continue
        for block in _iter_blocks(path):
            if _MAX_TASKS not in _header(block) or "forced_choice" not in block:
                continue
            task = _task(block)
            model_match = _MODEL_RE.search(block)
            overall_match = _OVERALL_RE.search(block)
            if task is None or model_match is None or overall_match is None:
                continue
            model_id = model_match.group(1).strip()
            params_b = parse_params_billions(model_id)
            if params_b is None:
                continue
            if is_olmo2_checkpoint_model_id(model_id):
                _, revision = parse_checkpoint_model_id(model_id)
                tokens_b = (
                    olmo2_tokens_billions_from_revision(revision)
                    if revision is not None
                    else None
                )
                if tokens_b in _EXCLUDED_OLMO_TOKEN_BUDGETS_B:
                    continue
            latest[(task, model_id)] = Result(
                task, model_id, params_b, float(overall_match.group(1))
            )
    return list(latest.values())


def _family(model_id: str) -> str | None:
    if is_pythia_checkpoint_model_id(model_id):
        return "pythia_checkpoints"
    if is_olmo2_checkpoint_model_id(model_id):
        return "olmo_checkpoints"
    if is_pythia_model(model_id):
        return "pythia"
    if is_olmo2_model(model_id):
        return "olmo"
    if is_qwen3_model(model_id):
        return "qwen3"
    if is_miniberta_model(model_id):
        return "miniberta"
    if is_babylm_model(model_id):
        return "babylm"
    return None


def _mean_line(points: list[Result]) -> list[tuple[float, float]]:
    """Mean duplicate models/seeds at each parameter count."""
    by_x: dict[float, list[float]] = {}
    for point in points:
        by_x.setdefault(point.params_b, []).append(point.accuracy)
    return sorted((x, sum(values) / len(values)) for x, values in by_x.items())


def _latest_checkpoint_line(points: list[Result]) -> list[tuple[float, float]]:
    """Highest-step checkpoint at each parameter count."""
    latest: dict[float, tuple[int, float]] = {}
    for point in points:
        step = checkpoint_step_from_model_id(point.model_id)
        if step is None:
            continue
        previous = latest.get(point.params_b)
        if previous is None or step > previous[0]:
            latest[point.params_b] = (step, point.accuracy)
    return sorted((x, accuracy) for x, (_, accuracy) in latest.items())


def plot(results: list[Result], *, output: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit("matplotlib required: pip install matplotlib") from exc

    fig = plt.figure(figsize=(18, 9))
    grid = GridSpec(2, 24, figure=fig, wspace=0.75, hspace=0.35)
    axes = [
        fig.add_subplot(grid[0, index * 6 : (index + 1) * 6])
        for index in range(4)
    ]
    axes.extend(
        fig.add_subplot(
            grid[1, 3 + index * 6 : 3 + (index + 1) * 6],
            sharex=axes[0],
            sharey=axes[0],
        )
        for index in range(3)
    )
    for ax in axes[1:4]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])

    grouped: dict[tuple[str, str], list[Result]] = {}
    for result in results:
        family = _family(result.model_id)
        if family is not None:
            grouped.setdefault((family, result.task), []).append(result)

    all_x = [result.params_b for result in results if _family(result.model_id)]
    for index, (ax, facet) in enumerate(zip(axes, _FACETS)):
        is_checkpoint = facet.endswith("_checkpoints")
        for task in _TASKS:
            points = grouped.get((facet, task), [])
            if not points:
                continue
            color = _TASK_COLORS[task]
            if is_checkpoint:
                ax.scatter(
                    [point.params_b for point in points],
                    [point.accuracy for point in points],
                    color=color,
                    alpha=0.18,
                    s=22,
                    edgecolors="none",
                    zorder=2,
                )
                line = _latest_checkpoint_line(points)
            else:
                ax.scatter(
                    [point.params_b for point in points],
                    [point.accuracy for point in points],
                    color=color,
                    alpha=0.32,
                    s=28,
                    edgecolors="none",
                    zorder=2,
                )
                line = _mean_line(points)
            if line:
                x, y = zip(*line)
                ax.plot(
                    x,
                    y,
                    "o-",
                    color=color,
                    linewidth=2.2,
                    markersize=6,
                    label=_TASK_LABELS[task],
                    zorder=3,
                )

        ax.set_xscale("log")
        ax.set_title(_FACET_LABELS[facet])
        ax.axhline(
            25,
            color="gray",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            zorder=1,
        )
        ax.axhline(
            50,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax.grid(True, which="both", alpha=0.25)
        if index >= 4:
            ax.set_xlabel("Model parameters (billions, log scale)")
        if index in (0, 4):
            ax.set_ylabel("Overall accuracy (%)")

    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        axes[0].set_xlim(xmin * 0.8, xmax * 1.25)
    axes[0].set_ylim(0, 100)

    handles = [
        Line2D(
            [0],
            [0],
            color=_TASK_COLORS[task],
            marker="o",
            linewidth=2.2,
            label=_TASK_LABELS[task],
        )
        for task in _TASKS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                alpha=0.3,
                linestyle="none",
                label="Individual run",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=1.1,
                label="Easy Ravens / Webb chance (25%)",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label="ABA chance (50%)",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument(
        "--babylm-experiments", type=Path, default=BABYLM_EXPERIMENTS_MD
    )
    parser.add_argument(
        "--pythia-checkpoints", type=Path, default=PYTHIA_CHECKPOINTS_MD
    )
    parser.add_argument("--olmo-checkpoints", type=Path, default=OLMO_CHECKPOINTS_MD)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    results = parse_latest(
        args.experiments,
        args.babylm_experiments,
        args.pythia_checkpoints,
        args.olmo_checkpoints,
    )
    output = args.output or (
        PLOTS_DIR / "faceted_task_model_parameter_scaling.png"
    )
    title = args.title or (
        "Forced-choice overall accuracy by model scale — most recent runs"
    )
    plot(results, output=output, title=title)


if __name__ == "__main__":
    main()
