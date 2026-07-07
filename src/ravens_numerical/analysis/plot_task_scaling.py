#!/usr/bin/env python3
"""Faceted per-task scaling curves from experiments.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.plot_utils import experiment_log_paths, iter_experiment_blocks
from ravens_numerical.models.registry import model_family, parse_params_billions
from ravens_numerical.paths import EXPERIMENTS_MD, PLOTS_DIR

_BY_TASK_RE = re.compile(
    r"\*\*By task:\*\* (.+)$", re.MULTILINE
)
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")

_FAMILIES = ("pythia", "qwen3", "babylm")
_FAMILY_STYLE = {
    "pythia": ("o-", "tab:blue", "Pythia"),
    "qwen3": ("s-", "tab:orange", "Qwen3"),
    "babylm": ("^-", "tab:green", "BabyLM"),
}


def parse_experiments_by_task(
    experiments_path: Path,
    *,
    date_prefix: str,
    prompt_type: str = "instruction",
    n_examples: int | None = None,
    prompt_mode: str | None = None,
    include_babylm_experiments: bool = True,
    families: tuple[str, ...] = _FAMILIES,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Return ``task_type -> {family -> [(params_b, accuracy_pct), ...]}``."""
    from ravens_numerical.analysis.experiment_log import TASK_TYPE_ORDER

    out: dict[str, dict[str, list[tuple[float, float]]]] = {
        tt: {f: [] for f in families} for tt in TASK_TYPE_ORDER
    }

    for block in iter_experiment_blocks(
        experiment_log_paths(
            experiments_path, include_babylm_experiments=include_babylm_experiments
        )
    ):
        header = block.split("\n", 1)[0]
        if date_prefix not in header:
            continue
        if f", {prompt_type})" not in header:
            continue
        if n_examples is not None and f"`--n-examples {n_examples}`" not in block:
            continue
        if prompt_mode is not None and f"`--ravens-prompt-mode {prompt_mode}`" not in block:
            continue

        model_match = re.search(r"^### (.+)$", block, re.MULTILINE)
        by_task_match = _BY_TASK_RE.search(block)
        if not model_match or not by_task_match:
            continue

        model_id = model_match.group(1).strip()
        params_b = parse_params_billions(model_id)
        family = model_family(model_id)
        if params_b is None or family not in families:
            continue

        for task_type, acc_str in _TASK_ACC_RE.findall(by_task_match.group(1)):
            if task_type not in out:
                out[task_type] = {f: [] for f in families}
            out[task_type][family].append((params_b, float(acc_str)))

    for task_type in out:
        for family in families:
            out[task_type][family].sort(key=lambda x: x[0])

    return out


def plot_task_scaling_facets(
    by_task: dict[str, dict[str, list[tuple[float, float]]]],
    *,
    output: Path,
    title: str,
    task_order: tuple[str, ...],
    families: tuple[str, ...] = _FAMILIES,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    n_tasks = len(task_order)
    ncols = 4
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False, sharex=True, sharey=True
    )

    for idx, task_type in enumerate(task_order):
        ax = axes[idx // ncols][idx % ncols]
        series = by_task.get(task_type, {f: [] for f in families})

        for family in families:
            points = series.get(family, [])
            if not points:
                continue
            fx, fy = zip(*points)
            fmt, color, label = _FAMILY_STYLE[family]
            ax.plot(fx, fy, fmt, color=color, linewidth=2, markersize=6, label=label)

        ax.set_title(task_type.replace("_", " "), fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(True, which="both", alpha=0.25)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Parameters (B, log scale)", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)

    for idx in range(n_tasks, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(families)),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=13, y=1.06)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    from ravens_numerical.analysis.experiment_log import TASK_TYPE_ORDER

    parser = argparse.ArgumentParser(description="Faceted per-task scaling from experiments.md")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=EXPERIMENTS_MD,
    )
    parser.add_argument("--date", default="2026-07-01")
    parser.add_argument(
        "--prompt-type",
        default="instruction",
        choices=("instruction", "completion"),
    )
    parser.add_argument("--n-examples", type=int, default=1)
    parser.add_argument("--prompt-mode", default="choice_only")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--no-babylm-experiments",
        action="store_true",
        help="Do not merge babyLMexperiments.md (Pythia/Qwen3 from experiments.md only)",
    )
    args = parser.parse_args()

    families: tuple[str, ...] = (
        ("pythia", "qwen3") if args.no_babylm_experiments else _FAMILIES
    )
    prompt_mode = args.prompt_mode if args.prompt_type == "instruction" else None
    by_task = parse_experiments_by_task(
        args.experiments,
        date_prefix=args.date,
        prompt_type=args.prompt_type,
        n_examples=args.n_examples,
        prompt_mode=prompt_mode,
        include_babylm_experiments=not args.no_babylm_experiments,
        families=families,
    )

    task_order = tuple(
        tt
        for tt in TASK_TYPE_ORDER
        if any(by_task.get(tt, {}).get(f) for f in families)
    )
    if not task_order:
        raise SystemExit(
            f"No matching entries in {args.experiments} "
            f"(date={args.date}, prompt_type={args.prompt_type})"
        )

    output = args.output or (
        PLOTS_DIR / f"task_scaling_{args.prompt_type}_{args.date}_n{args.n_examples}.png"
    )
    title = args.title or (
        f"Instruction choice-only — per-task accuracy by scale "
        f"(n_examples={args.n_examples}, {args.date}, 20 tasks each)"
    )
    plot_task_scaling_facets(
        by_task, output=output, title=title, task_order=task_order, families=families
    )


if __name__ == "__main__":
    main()
