#!/usr/bin/env python3
"""BabyLM 10M / 100M on the three challenge subtasks, by SFT mix.

Conditions (forced-choice, n=0, complete 500-item eval unless noted):

- Base Hub checkpoints
- SFT on the original 7 types (``not_challenge``)
- SFT on the 3 challenge types only (``only_challenge``)
- SFT on all 10 types (``complete_all_types`` / ``complete-all-types``)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.dump_scores import overlay_log_scores
from ravens_numerical.generation.generator import CHALLENGE_TYPE_CYCLE
from ravens_numerical.paths import BABYLM_FINETUNE_LOGS_DIR, PLOTS_DIR

_CHANCE_PCT = 25.0
_SCALES = ("10m", "100m")
_SCALE_TITLES = {"10m": "BabyLM 10M", "100m": "BabyLM 100M"}
_TASK_LABELS = {
    "distribution_of_three": "Distribution\nof three",
    "progression_plus_n": "Progression\n+n",
    "tuple_grid": "Tuple grid",
}
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")
_BY_TASK_RE = re.compile(r"\*\*By task:\*\* (.+)$", re.MULTILINE)

# condition_key, legend label, run_id or Hub id per scale
_CONDITIONS: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "base",
        "Base",
        {
            "10m": "BabyLM-community/babylm-baseline-10m-gpt2",
            "100m": "BabyLM-community/babylm-baseline-100m-gpt2",
        },
    ),
    (
        "sft7",
        "SFT on 7",
        {
            "10m": "babylm-10m-gpt2__not_challenge__20260814T210501Z",
            "100m": "babylm-100m-gpt2__not_challenge__20260814T210306Z",
        },
    ),
    (
        "sft3",
        "SFT on 3 new",
        {
            "10m": "babylm-10m-gpt2__only_challenge__20260814T204510Z",
            "100m": "babylm-100m-gpt2__only_challenge__20260814T204519Z",
        },
    ),
    (
        "sft10",
        "SFT on all 10",
        {
            "10m": "babylm-10m-gpt2__complete-all-types__20260814T200911Z",
            "100m": "babylm-100m-gpt2__complete_all_types__20260814T200850Z",
        },
    ),
)
_CONDITION_COLORS = {
    "base": "#4c78a8",
    "sft7": "#f58518",
    "sft3": "#54a24b",
    "sft10": "#e45756",
}


def _parse_by_task_from_log(run_id: str) -> dict[str, float]:
    path = BABYLM_FINETUNE_LOGS_DIR / f"{run_id}__n0.md"
    if not path.is_file():
        path = BABYLM_FINETUNE_LOGS_DIR / f"{run_id}.md"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    match = None
    for match in _BY_TASK_RE.finditer(text):
        pass
    if match is None:
        return {}
    return {task: float(acc) for task, acc in _TASK_ACC_RE.findall(match.group(1))}


def load_condition_tasks(query: str) -> dict[str, float]:
    """Challenge-subtask accuracies from dump overlay, else last log By-task line."""
    logged = _parse_by_task_from_log(query)
    _overall, by_task = overlay_log_scores(query, "challenge", None, logged)
    return {
        task: float(by_task[task])
        for task in CHALLENGE_TYPE_CYCLE
        if task in by_task
    }


def collect() -> dict[str, dict[str, dict[str, float]]]:
    """``scale -> condition_key -> {task: acc}``."""
    out: dict[str, dict[str, dict[str, float]]] = {s: {} for s in _SCALES}
    for key, _label, ids in _CONDITIONS:
        for scale, query in ids.items():
            tasks = load_condition_tasks(query)
            if len(tasks) != len(CHALLENGE_TYPE_CYCLE):
                missing = [t for t in CHALLENGE_TYPE_CYCLE if t not in tasks]
                raise SystemExit(
                    f"missing challenge subtasks {missing} for {scale} {key} ({query})"
                )
            out[scale][key] = tasks
    return out


def plot(
    data: dict[str, dict[str, dict[str, float]]],
    *,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required") from exc

    tasks = list(CHALLENGE_TYPE_CYCLE)
    n_cond = len(_CONDITIONS)
    x = np.arange(len(tasks))
    width = 0.18
    offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * width

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, scale in zip(axes, _SCALES):
        for offset, (key, _label, _ids) in zip(offsets, _CONDITIONS):
            values = [data[scale][key][task] for task in tasks]
            ax.bar(
                x + offset,
                values,
                width=width,
                color=_CONDITION_COLORS[key],
                edgecolor="black",
                linewidth=0.4,
                zorder=2,
            )
        ax.axhline(
            _CHANCE_PCT,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            zorder=1,
        )
        ax.set_title(_SCALE_TITLES[scale])
        ax.set_xticks(x)
        ax.set_xticklabels([_TASK_LABELS[t] for t in tasks], fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy (%)")
    handles = [
        Patch(facecolor=_CONDITION_COLORS[key], edgecolor="black", label=label)
        for key, label, _ids in _CONDITIONS
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Chance ({_CHANCE_PCT:g}%)",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=PLOTS_DIR / "babylm_challenge_sft_conditions.png",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    data = collect()
    for scale in _SCALES:
        print(f"{_SCALE_TITLES[scale]}:")
        for key, label, _ids in _CONDITIONS:
            tasks = data[scale][key]
            mean = sum(tasks.values()) / len(tasks)
            parts = "  ".join(
                f"{t.split('_')[0][:4]}={tasks[t]:5.1f}" for t in CHALLENGE_TYPE_CYCLE
            )
            print(f"  {label:16s} mean={mean:5.1f}  {parts}")

    title = args.title or (
        "Challenge subtasks — BabyLM base vs SFT mix "
        "(forced choice, n=0)"
    )
    plot(data, output=args.output.resolve(), title=title)


if __name__ == "__main__":
    main()
