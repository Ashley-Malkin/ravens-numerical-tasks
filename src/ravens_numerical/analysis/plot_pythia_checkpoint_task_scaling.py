#!/usr/bin/env python3
"""Faceted Pythia completion accuracy vs training tokens (checkpoint sweep).

One panel per parameter size (70M, 160M, 1B).  X-axis is tokens trained on
(``step × 2,097,152`` from ``pythiacheckpoints.md`` checkpoints), not step
count.  Supports per-subtask lines or overall accuracy.  A final star marks
the fully-trained model (~300B tokens) from 2026-07-01 ``experiments.md``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.experiment_log import TASK_TYPE_ORDER
from ravens_numerical.models.registry import (
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    PYTHIA_FULL_TRAINING_TOKENS_B,
    base_model_id,
    is_pythia_checkpoint_model_id,
    parse_pythia_checkpoint_model_id,
    pythia_step_from_revision,
    pythia_tokens_billions_at_step,
)
from ravens_numerical.paths import EXPERIMENTS_MD, PLOTS_DIR, PYTHIA_CHECKPOINTS_MD

_FINAL_DATE = "2026-07-01"
_CHECKPOINT_MAX_TASKS = "(max_tasks=140)"
_FINAL_MAX_TASKS = "(max_tasks=140)"

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")
_BY_TASK_RE = re.compile(r"\*\*By task:\*\* (.+)$", re.MULTILINE)
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")

_FACET_LABEL = {
    "EleutherAI/pythia-70m-deduped": "Pythia 70M",
    "EleutherAI/pythia-160m-deduped": "Pythia 160M",
    "EleutherAI/pythia-1b-deduped": "Pythia 1B",
}
_MODEL_COLORS = {
    "EleutherAI/pythia-70m-deduped": "tab:blue",
    "EleutherAI/pythia-160m-deduped": "tab:orange",
    "EleutherAI/pythia-1b-deduped": "tab:green",
}


def _iter_blocks(path: Path):
    return re.split(r"\n---\n", path.read_text(encoding="utf-8"))


def _parse_by_task(block: str) -> dict[str, float]:
    match = _BY_TASK_RE.search(block)
    if not match:
        return {}
    return {
        task: float(acc)
        for task, acc in _TASK_ACC_RE.findall(match.group(1))
    }


def parse_checkpoint_by_task(
    checkpoints_path: Path,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """``base_model_id -> {task_type -> [(tokens_b, accuracy_pct), ...]}``."""
    out: dict[str, dict[str, list[tuple[float, float]]]] = {
        base: {tt: [] for tt in TASK_TYPE_ORDER}
        for base in PYTHIA_CHECKPOINT_SCALING_MODELS
    }

    for block in _iter_blocks(checkpoints_path):
        header = block.split("\n", 1)[0]
        if _CHECKPOINT_MAX_TASKS not in header or ", completion)" not in header:
            continue
        if "`--n-examples 1`" not in block:
            continue

        model_match = _MODEL_RE.search(block)
        if not model_match:
            continue
        model_id = model_match.group(1).strip()
        if not is_pythia_checkpoint_model_id(model_id):
            continue

        base = base_model_id(model_id)
        if base not in out:
            continue
        _, revision = parse_pythia_checkpoint_model_id(model_id)
        if revision is None:
            continue
        step = pythia_step_from_revision(revision)
        if step is None:
            continue
        tokens_b = pythia_tokens_billions_at_step(step)

        by_task = _parse_by_task(block)
        for task_type, acc in by_task.items():
            if task_type not in out[base]:
                out[base][task_type] = []
            out[base][task_type].append((tokens_b, acc))

    for base in out:
        for task_type in out[base]:
            out[base][task_type].sort(key=lambda x: x[0])
    return out


def parse_final_by_task(
    experiments_path: Path,
    *,
    date_prefix: str = _FINAL_DATE,
) -> dict[str, dict[str, float]]:
    """``base_model_id -> {task_type -> accuracy_pct}`` for full Pythia models."""
    out: dict[str, dict[str, float]] = {}

    for block in _iter_blocks(experiments_path):
        header = block.split("\n", 1)[0]
        if date_prefix not in header or _FINAL_MAX_TASKS not in header:
            continue
        if ", completion)" not in header or "`--n-examples 1`" not in block:
            continue

        model_match = _MODEL_RE.search(block)
        if not model_match:
            continue
        model_id = model_match.group(1).strip()
        if is_pythia_checkpoint_model_id(model_id):
            continue
        base = base_model_id(model_id)
        if base not in PYTHIA_CHECKPOINT_SCALING_MODELS:
            continue

        by_task = _parse_by_task(block)
        if by_task:
            out[base] = by_task

    return out


def parse_checkpoint_overall(
    checkpoints_path: Path,
) -> dict[str, list[tuple[float, float]]]:
    """``base_model_id -> [(tokens_b, overall_accuracy_pct), ...]``."""
    out: dict[str, list[tuple[float, float]]] = {
        base: [] for base in PYTHIA_CHECKPOINT_SCALING_MODELS
    }

    for block in _iter_blocks(checkpoints_path):
        header = block.split("\n", 1)[0]
        if _CHECKPOINT_MAX_TASKS not in header or ", completion)" not in header:
            continue
        if "`--n-examples 1`" not in block:
            continue

        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if not is_pythia_checkpoint_model_id(model_id):
            continue

        base = base_model_id(model_id)
        if base not in out:
            continue
        _, revision = parse_pythia_checkpoint_model_id(model_id)
        if revision is None:
            continue
        step = pythia_step_from_revision(revision)
        if step is None:
            continue
        out[base].append(
            (pythia_tokens_billions_at_step(step), float(overall_match.group(1)))
        )

    for base in out:
        out[base].sort(key=lambda x: x[0])
    return out


def parse_final_overall(
    experiments_path: Path,
    *,
    date_prefix: str = _FINAL_DATE,
) -> dict[str, float]:
    """``base_model_id -> overall_accuracy_pct`` for full Pythia models."""
    out: dict[str, float] = {}

    for block in _iter_blocks(experiments_path):
        header = block.split("\n", 1)[0]
        if date_prefix not in header or _FINAL_MAX_TASKS not in header:
            continue
        if ", completion)" not in header or "`--n-examples 1`" not in block:
            continue

        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if is_pythia_checkpoint_model_id(model_id):
            continue
        base = base_model_id(model_id)
        if base not in PYTHIA_CHECKPOINT_SCALING_MODELS:
            continue
        out[base] = float(overall_match.group(1))

    return out


def _task_colors(task_order: tuple[str, ...], cmap_name: str = "tab10") -> dict[str, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    n = len(task_order)
    return {
        task: cmap(i / max(1, n - 1) if n > 1 else 0.0)
        for i, task in enumerate(task_order)
    }


def plot_pythia_checkpoint_facets(
    *,
    checkpoints: dict[str, dict[str, list[tuple[float, float]]]],
    finals: dict[str, dict[str, float]],
    task_order: tuple[str, ...],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    ncols = len(PYTHIA_CHECKPOINT_SCALING_MODELS)
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.5 * ncols, 5), squeeze=False, sharey=True
    )
    colors = _task_colors(task_order)

    for idx, base in enumerate(PYTHIA_CHECKPOINT_SCALING_MODELS):
        ax = axes[0][idx]
        series = checkpoints.get(base, {})
        final = finals.get(base, {})

        for task_type in task_order:
            points = series.get(task_type, [])
            if points:
                tx, ty = zip(*points)
                ax.plot(
                    tx,
                    ty,
                    "o-",
                    color=colors[task_type],
                    linewidth=2,
                    markersize=6,
                    label=task_type.replace("_", " "),
                )
            if task_type in final:
                ax.plot(
                    PYTHIA_FULL_TRAINING_TOKENS_B,
                    final[task_type],
                    marker="*",
                    markersize=14,
                    color=colors[task_type],
                    linestyle="none",
                    zorder=4,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tokens trained (billions, log scale)")
        ax.set_title(_FACET_LABEL.get(base, base))
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")

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
    extra_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="none",
            markersize=12,
            label=f"Full model (~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B tokens, Jul 1)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Chance (25%)",
        ),
    ]
    fig.legend(
        handles=task_handles + extra_handles,
        loc="upper center",
        ncol=min(4, len(task_order) + 2),
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def plot_pythia_checkpoint_overall_facets(
    *,
    checkpoints: dict[str, list[tuple[float, float]]],
    finals: dict[str, float],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    ncols = len(PYTHIA_CHECKPOINT_SCALING_MODELS)
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.5 * ncols, 5), squeeze=False, sharey=True
    )

    for idx, base in enumerate(PYTHIA_CHECKPOINT_SCALING_MODELS):
        ax = axes[0][idx]
        points = checkpoints.get(base, [])
        if points:
            tx, ty = zip(*points)
            ax.plot(tx, ty, "o-", color="tab:blue", linewidth=2, markersize=8)
        if base in finals:
            ax.plot(
                PYTHIA_FULL_TRAINING_TOKENS_B,
                finals[base],
                marker="*",
                markersize=16,
                color="tab:blue",
                linestyle="none",
                zorder=4,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Tokens trained (billions, log scale)")
        ax.set_title(_FACET_LABEL.get(base, base))
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Overall accuracy (%)")

    handles = [
        Line2D([0], [0], color="tab:blue", marker="o", linewidth=2, label="Checkpoints"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="tab:blue",
            linestyle="none",
            markersize=12,
            label=f"Full model (~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B tokens, Jul 1)",
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Chance (25%)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def plot_pythia_checkpoint_overall_combined(
    *,
    checkpoints: dict[str, list[tuple[float, float]]],
    finals: dict[str, float],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))

    for base in PYTHIA_CHECKPOINT_SCALING_MODELS:
        color = _MODEL_COLORS[base]
        label = _FACET_LABEL[base]
        points = checkpoints.get(base, [])
        if points:
            tx, ty = zip(*points)
            ax.plot(
                tx,
                ty,
                "o-",
                color=color,
                linewidth=2,
                markersize=8,
                label=label,
            )
        if base in finals:
            ax.plot(
                PYTHIA_FULL_TRAINING_TOKENS_B,
                finals[base],
                marker="*",
                markersize=16,
                color=color,
                linestyle="none",
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Tokens trained (billions, log scale)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(title)
    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance (25%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def plot_pythia_checkpoint_by_task_facets(
    *,
    checkpoints: dict[str, dict[str, list[tuple[float, float]]]],
    finals: dict[str, dict[str, float]],
    task_order: tuple[str, ...],
    output: Path,
    title: str,
) -> None:
    """One panel per subtask; colored lines for 70M / 160M / 1B."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    n_tasks = len(task_order)
    ncols = 4
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False, sharex=True, sharey=True
    )

    for idx, task_type in enumerate(task_order):
        ax = axes[idx // ncols][idx % ncols]
        for base in PYTHIA_CHECKPOINT_SCALING_MODELS:
            color = _MODEL_COLORS[base]
            label = _FACET_LABEL[base]
            points = checkpoints.get(base, {}).get(task_type, [])
            if points:
                tx, ty = zip(*points)
                ax.plot(
                    tx,
                    ty,
                    "o-",
                    color=color,
                    linewidth=2,
                    markersize=5,
                    label=label,
                )
            final = finals.get(base, {})
            if task_type in final:
                ax.plot(
                    PYTHIA_FULL_TRAINING_TOKENS_B,
                    final[task_type],
                    marker="*",
                    markersize=12,
                    color=color,
                    linestyle="none",
                    zorder=4,
                )

        ax.set_title(task_type.replace("_", " "), fontsize=11)
        ax.set_xscale("log")
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.25)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Tokens trained (B, log scale)", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)

    for idx in range(n_tasks, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    model_handles = [
        Line2D(
            [0],
            [0],
            color=_MODEL_COLORS[base],
            marker="o",
            linewidth=2,
            label=_FACET_LABEL[base],
        )
        for base in PYTHIA_CHECKPOINT_SCALING_MODELS
    ]
    extra_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="none",
            markersize=10,
            label=f"Full model (~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B tokens, Jul 1)",
        ),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Chance (25%)"),
    ]
    fig.legend(
        handles=model_handles + extra_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, default=PYTHIA_CHECKPOINTS_MD
    )
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument(
        "--metric",
        choices=("task", "task-by-task", "overall", "overall-combined", "both", "all"),
        default="both",
        help=(
            "Plot per-task facets by model (task), by task (task-by-task), "
            "overall facets, combined overall lines, task+overall facets (both), "
            "or all plots (all)"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--task-by-task-output", type=Path, default=None)
    parser.add_argument("--overall-output", type=Path, default=None)
    parser.add_argument("--combined-output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--task-by-task-title", default=None)
    parser.add_argument("--overall-title", default=None)
    parser.add_argument("--combined-title", default=None)
    args = parser.parse_args()

    metrics = {args.metric}
    if args.metric == "both":
        metrics = {"task", "overall"}
    elif args.metric == "all":
        metrics = {"task", "task-by-task", "overall", "overall-combined"}

    needs_task_data = "task" in metrics or "task-by-task" in metrics
    if needs_task_data:
        checkpoints = parse_checkpoint_by_task(args.checkpoints)
        finals = parse_final_by_task(args.experiments)

        task_order = tuple(
            tt
            for tt in TASK_TYPE_ORDER
            if any(checkpoints.get(base, {}).get(tt) for base in PYTHIA_CHECKPOINT_SCALING_MODELS)
            or any(
                finals.get(base, {}).get(tt) is not None
                for base in PYTHIA_CHECKPOINT_SCALING_MODELS
            )
        )
        if not task_order:
            raise SystemExit(f"No checkpoint or final Pythia completion data found")

    if "task" in metrics:
        output = args.output or (PLOTS_DIR / "pythia_checkpoint_task_scaling_n1.png")
        title = args.title or (
            "Pythia completion — per-task accuracy vs tokens trained "
            f"(n=1; faceted by model; full ~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B)"
        )
        plot_pythia_checkpoint_facets(
            checkpoints=checkpoints,
            finals=finals,
            task_order=task_order,
            output=output,
            title=title,
        )

    if "task-by-task" in metrics:
        by_task_output = args.task_by_task_output or (
            PLOTS_DIR / "pythia_checkpoint_task_by_task_scaling_n1.png"
        )
        by_task_title = args.task_by_task_title or (
            "Pythia completion — per-task accuracy vs tokens trained "
            f"(n=1; faceted by task; full ~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B)"
        )
        plot_pythia_checkpoint_by_task_facets(
            checkpoints=checkpoints,
            finals=finals,
            task_order=task_order,
            output=by_task_output,
            title=by_task_title,
        )

    if "overall" in metrics or "overall-combined" in metrics:
        overall_checkpoints = parse_checkpoint_overall(args.checkpoints)
        overall_finals = parse_final_overall(args.experiments)
        if not any(overall_checkpoints.values()) and not overall_finals:
            raise SystemExit("No checkpoint or final Pythia overall completion data found")

    if "overall" in metrics:
        overall_output = args.overall_output or (
            PLOTS_DIR / "pythia_checkpoint_overall_scaling_n1.png"
        )
        overall_title = args.overall_title or (
            "Pythia completion — overall accuracy vs tokens trained "
            f"(n=1; checkpoints + full ~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B)"
        )
        plot_pythia_checkpoint_overall_facets(
            checkpoints=overall_checkpoints,
            finals=overall_finals,
            output=overall_output,
            title=overall_title,
        )

    if "overall-combined" in metrics:
        combined_output = args.combined_output or (
            PLOTS_DIR / "pythia_checkpoint_overall_combined_n1.png"
        )
        combined_title = args.combined_title or (
            "Pythia completion — overall accuracy vs tokens trained "
            f"(n=1; 70M / 160M / 1B; full ~{PYTHIA_FULL_TRAINING_TOKENS_B:.0f}B)"
        )
        plot_pythia_checkpoint_overall_combined(
            checkpoints=overall_checkpoints,
            finals=overall_finals,
            output=combined_output,
            title=combined_title,
        )


if __name__ == "__main__":
    main()
