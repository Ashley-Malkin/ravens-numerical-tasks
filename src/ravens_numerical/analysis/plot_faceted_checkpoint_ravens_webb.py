#!/usr/bin/env python3
"""Compare Easy Ravens, Webb, and ABA accuracy over training tokens.

The two facets show Pythia and OLMo 2 checkpoint trajectories plus full-model
stars. Data are the latest matching forced-choice runs for each task and model
with ``max_tasks=140``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.models.registry import (
    OLMO2_CHECKPOINT_SCALING_MODELS,
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    PYTHIA_FULL_TRAINING_TOKENS_B,
    base_model_id,
    is_olmo2_base_model,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    is_pythia_model,
    olmo2_tokens_billions_from_revision,
    parse_checkpoint_model_id,
    pythia_step_from_revision,
    pythia_tokens_billions_at_step,
)
from ravens_numerical.paths import (
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PLOTS_DIR,
    PYTHIA_CHECKPOINTS_MD,
)

_TASKS = ("ravens", "webb", "aba")
_TASK_LABELS = {"ravens": "Easy Ravens", "webb": "Webb", "aba": "ABA"}
_MAX_TASKS = "(max_tasks=140)"
_EXCLUDED_OLMO_TOKEN_BUDGETS_B = {51.0}
_MIN_LOG_X_B = 0.0005

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")

_PYTHIA_LABELS = {
    "EleutherAI/pythia-70m-deduped": "70M",
    "EleutherAI/pythia-160m-deduped": "160M",
    "EleutherAI/pythia-1b-deduped": "1B",
}
_OLMO_LABELS = {
    "allenai/OLMo-2-0425-1B": "1B",
    "allenai/OLMo2-7B-1124": "7B",
    "allenai/OLMo-2-13B-1124": "13B",
}
_OLMO_FULL_TOKENS_B = {
    "allenai/OLMo-2-0425-1B": 4000.0,
    "allenai/OLMo2-7B-1124": 3900.0,
    "allenai/OLMo-2-13B-1124": 3900.0,
}


def _iter_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    yield from (
        part.strip()
        for part in re.split(
            r"(?=^## \d{4}-\d{2}-\d{2} )", text, flags=re.MULTILINE
        )
        if part.strip().startswith("## ")
    )


def _header(block: str) -> str:
    return next((line for line in block.splitlines() if line.strip()), "")


def _matches(block: str, task: str) -> bool:
    header = _header(block)
    return (
        _MAX_TASKS in header
        and "forced_choice" in block
        and f"task-type {task}" in block
    )


def parse_latest(*paths: Path) -> dict[str, dict[str, float]]:
    """Return ``task -> model_id -> latest overall accuracy percent``."""
    latest = {task: {} for task in _TASKS}
    for path in paths:
        if not path.is_file():
            continue
        for block in _iter_blocks(path):
            model_match = _MODEL_RE.search(block)
            overall_match = _OVERALL_RE.search(block)
            if not model_match or not overall_match:
                continue
            for task in _TASKS:
                if _matches(block, task):
                    latest[task][model_match.group(1).strip()] = float(
                        overall_match.group(1)
                    )
                    break
    return latest


def _checkpoint_tokens_b(model_id: str) -> float | None:
    _, revision = parse_checkpoint_model_id(model_id)
    if revision is None:
        return None
    if is_pythia_checkpoint_model_id(model_id):
        step = pythia_step_from_revision(revision)
        return pythia_tokens_billions_at_step(step) if step is not None else None
    if is_olmo2_checkpoint_model_id(model_id):
        tokens_b = olmo2_tokens_billions_from_revision(revision)
        if (
            tokens_b is None
            or tokens_b <= 0
            or tokens_b in _EXCLUDED_OLMO_TOKEN_BUDGETS_B
        ):
            return None
        return tokens_b
    return None


def checkpoint_series(
    latest: dict[str, dict[str, float]],
    *,
    family: str,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Return ``task -> base model -> [(tokens_b, accuracy), ...]``."""
    bases = (
        PYTHIA_CHECKPOINT_SCALING_MODELS
        if family == "pythia"
        else OLMO2_CHECKPOINT_SCALING_MODELS
    )
    out = {task: {base: [] for base in bases} for task in _TASKS}
    for task, results in latest.items():
        for model_id, accuracy in results.items():
            is_checkpoint = (
                is_pythia_checkpoint_model_id(model_id)
                if family == "pythia"
                else is_olmo2_checkpoint_model_id(model_id)
            )
            if not is_checkpoint:
                continue
            base = base_model_id(model_id)
            if base not in out[task]:
                continue
            tokens_b = _checkpoint_tokens_b(model_id)
            if tokens_b is not None:
                out[task][base].append((tokens_b, accuracy))
    for by_model in out.values():
        for points in by_model.values():
            points.sort()
    return out


def full_models(
    latest: dict[str, dict[str, float]],
    *,
    family: str,
) -> dict[str, dict[str, float]]:
    """Return ``task -> base model -> full-model accuracy``."""
    bases = (
        PYTHIA_CHECKPOINT_SCALING_MODELS
        if family == "pythia"
        else OLMO2_CHECKPOINT_SCALING_MODELS
    )
    out = {task: {} for task in _TASKS}
    for task, results in latest.items():
        for model_id, accuracy in results.items():
            if "@" in model_id or model_id not in bases:
                continue
            if family == "pythia" and is_pythia_model(model_id):
                out[task][model_id] = accuracy
            elif family == "olmo" and is_olmo2_base_model(model_id):
                out[task][model_id] = accuracy
    return out


def _shades(cmap, count: int) -> list:
    """Dark-to-light shades, ordered from smallest to largest model."""
    if count == 1:
        return [cmap(0.75)]
    return [cmap(0.85 - index * 0.35 / (count - 1)) for index in range(count)]


def plot(
    latest: dict[str, dict[str, float]],
    *,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit("matplotlib required: pip install matplotlib") from exc

    pythia_series = checkpoint_series(latest, family="pythia")
    olmo_series = checkpoint_series(latest, family="olmo")
    pythia_full = full_models(latest, family="pythia")
    olmo_full = full_models(latest, family="olmo")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            "Pythia",
            PYTHIA_CHECKPOINT_SCALING_MODELS,
            _PYTHIA_LABELS,
            pythia_series,
            pythia_full,
            {base: PYTHIA_FULL_TRAINING_TOKENS_B for base in _PYTHIA_LABELS},
        ),
        (
            axes[1],
            "OLMo 2",
            OLMO2_CHECKPOINT_SCALING_MODELS,
            _OLMO_LABELS,
            olmo_series,
            olmo_full,
            _OLMO_FULL_TOKENS_B,
        ),
    )

    task_cmaps = {
        "ravens": plt.get_cmap("Greens"),
        "webb": plt.get_cmap("Blues"),
        "aba": plt.get_cmap("Purples"),
    }
    all_x: list[float] = []
    legend_handles: list = []

    for ax, family_label, bases, labels, series, finals, full_tokens in panels:
        colors = {
            task: dict(zip(bases, _shades(task_cmaps[task], len(bases))))
            for task in _TASKS
        }
        for task in _TASKS:
            for base in bases:
                points = series[task][base]
                color = colors[task][base]
                if points:
                    x, y = zip(*points)
                    ax.plot(
                        x,
                        y,
                        "o-",
                        color=color,
                        linewidth=2,
                        markersize=6,
                    )
                    all_x.extend(x)
                if base in finals[task]:
                    ax.plot(
                        full_tokens[base],
                        finals[task][base],
                        marker="*",
                        color=color,
                        markersize=14,
                        linestyle="none",
                        zorder=4,
                    )
                    all_x.append(full_tokens[base])
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker="o",
                        linewidth=2,
                        label=(
                            f"{family_label} {labels[base]} — "
                            f"{_TASK_LABELS[task]}"
                        ),
                    )
                )

        ax.set_title(family_label)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens seen (billions, log scale)")
        ax.axhline(25, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(50, color="gray", linestyle=":", linewidth=1.1, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Overall accuracy (%)")
    if all_x:
        axes[0].set_xlim(max(_MIN_LOG_X_B, min(all_x) * 0.75), max(all_x) * 1.15)

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                linestyle="none",
                markersize=12,
                label="Full model",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=1,
                label="Easy Ravens / Webb chance (25%)",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle=":",
                linewidth=1.1,
                label="ABA chance (50%)",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.89, bottom=0.34, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument(
        "--pythia-checkpoints", type=Path, default=PYTHIA_CHECKPOINTS_MD
    )
    parser.add_argument("--olmo-checkpoints", type=Path, default=OLMO_CHECKPOINTS_MD)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    latest = parse_latest(
        args.experiments, args.pythia_checkpoints, args.olmo_checkpoints
    )
    output = args.output or (
        PLOTS_DIR
        / "faceted_checkpoints_easy_ravens_webb_aba_forced_choice_scaling.png"
    )
    title = args.title or (
        "Easy Ravens, Webb, and ABA accuracy vs tokens seen — "
        "forced choice, max_tasks=140"
    )
    plot(latest, output=output, title=title)


if __name__ == "__main__":
    main()
