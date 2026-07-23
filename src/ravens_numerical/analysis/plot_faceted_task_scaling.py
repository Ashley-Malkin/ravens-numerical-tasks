#!/usr/bin/env python3
"""Faceted per-subtask scaling: one panel per model × prompt type.

Each facet plots a labeled line for every task type (combine, constancy, …)
using n=1 runs from 2026-07-01 — the same date as the n=1 curves in the
overall faceted ICL plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ravens_numerical.analysis.experiment_log import TASK_TYPE_ORDER
from ravens_numerical.analysis.plot_qwen3_instruction_icl import parse_by_task_icl
from ravens_numerical.paths import EXPERIMENTS_MD, PLOTS_DIR

_N1_DATE = {1: "2026-07-01"}

_FACETS: tuple[tuple[str, str], ...] = (
    ("pythia", "completion"),
    ("qwen3", "completion"),
    ("pythia", "instruction"),
    ("qwen3", "instruction"),
)
_FACET_LABEL = {
    ("pythia", "completion"): "Pythia completion",
    ("qwen3", "completion"): "Qwen3 completion",
    ("pythia", "instruction"): "Pythia instruction",
    ("qwen3", "instruction"): "Qwen3 instruction",
}


def _task_colors(task_order: tuple[str, ...], cmap_name: str = "tab10") -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    n = len(task_order)
    return {
        task: cmap(i / max(1, n - 1) if n > 1 else 0.0)
        for i, task in enumerate(task_order)
    }


def plot_faceted_by_model_prompt(
    *,
    series: dict[tuple[str, str], dict[str, dict[int, list[tuple[float, float]]]]],
    task_order: tuple[str, ...],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    ncols = 2
    nrows = (len(_FACETS) + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False, sharey=True
    )
    colors = _task_colors(task_order)

    for idx, (family, prompt_type) in enumerate(_FACETS):
        ax = axes_grid[idx // ncols][idx % ncols]
        by_task = series.get((family, prompt_type), {})

        for task_type in task_order:
            points = by_task.get(task_type, {}).get(1, [])
            if not points:
                continue
            px, py = zip(*points)
            ax.plot(
                px,
                py,
                "o-",
                color=colors[task_type],
                linewidth=2,
                markersize=6,
                label=task_type.replace("_", " "),
            )

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_title(_FACET_LABEL[(family, prompt_type)])
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)")

    for idx in range(len(_FACETS), nrows * ncols):
        axes_grid[idx // ncols][idx % ncols].axis("off")

    task_handles = [
        Line2D(
            [0],
            [0],
            color=colors[t],
            marker="o",
            linewidth=2,
            label=t.replace("_", " "),
        )
        for t in task_order
    ]
    chance_handle = Line2D(
        [0], [0], color="gray", linestyle="--", linewidth=1, label="Chance (25%)"
    )
    fig.legend(
        handles=task_handles + [chance_handle],
        loc="upper center",
        ncol=min(4, len(task_order) + 1),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    series: dict[tuple[str, str], dict[str, dict[int, list[tuple[float, float]]]]] = {}
    for family, prompt_type in _FACETS:
        prompt_mode = "choice_only" if prompt_type == "instruction" else None
        series[(family, prompt_type)] = parse_by_task_icl(
            args.experiments,
            family=family,
            n_example_dates=_N1_DATE,
            prompt_type=prompt_type,
            prompt_mode=prompt_mode,
        )

    task_order = tuple(
        tt
        for tt in TASK_TYPE_ORDER
        if any(
            series.get((family, prompt_type), {}).get(tt, {}).get(1)
            for family, prompt_type in _FACETS
        )
    )
    if not task_order:
        raise SystemExit(
            f"No n=1 (2026-07-01) per-task entries in {args.experiments}"
        )

    output = args.output or (PLOTS_DIR / "faceted_task_scaling_n1.png")
    title = args.title or (
        "Per-task accuracy by scale (n=1, 2026-07-01) — faceted by model and prompt type"
    )
    plot_faceted_by_model_prompt(
        series=series,
        task_order=task_order,
        output=output,
        title=title,
    )


if __name__ == "__main__":
    main()
