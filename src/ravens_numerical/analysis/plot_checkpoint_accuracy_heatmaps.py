#!/usr/bin/env python3
"""Parameter × training-token heatmaps for checkpoint task accuracy.

Produces a 2×3 figure:

- rows: Pythia and OLMo 2
- columns: Easy Ravens, Webb, and ABA

Rows within each heatmap are model parameter sizes, columns are tokens seen,
and cell color/text show the latest forced-choice overall accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ravens_numerical.analysis.plot_faceted_checkpoint_ravens_webb import (
    _checkpoint_tokens_b,
    full_models,
    parse_latest,
)
from ravens_numerical.models.registry import (
    OLMO2_CHECKPOINT_SCALING_MODELS,
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    base_model_id,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    parse_params_billions,
)
from ravens_numerical.paths import (
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PLOTS_DIR,
    PYTHIA_CHECKPOINTS_MD,
)

_TASKS = ("ravens", "webb", "aba")
_TASK_LABELS = {"ravens": "Easy Ravens", "webb": "Webb", "aba": "ABA"}
_TASK_CHANCE = {"ravens": 25.0, "webb": 25.0, "aba": 50.0}
_FAMILIES = ("pythia", "olmo")
_FAMILY_LABELS = {"pythia": "Pythia", "olmo": "OLMo 2"}
_FULL_TOKEN_LABELS = {
    "pythia": "Full\n(300B tokens)",
    "olmo": "Full\n(~4T tokens)",
}


def _format_params(params_b: float) -> str:
    if params_b < 1:
        return f"{params_b * 1000:g}M"
    return f"{params_b:g}B"


def _format_tokens(tokens_b: float) -> str:
    if tokens_b < 1:
        return f"{tokens_b:.2g}B"
    return f"{tokens_b:g}B"


def build_matrix(
    latest: dict[str, dict[str, float]],
    *,
    family: str,
    task: str,
) -> tuple[list[float], list[float], list[list[float | None]]]:
    """Return parameter rows, token columns, and sparse accuracy matrix."""
    bases = (
        PYTHIA_CHECKPOINT_SCALING_MODELS
        if family == "pythia"
        else OLMO2_CHECKPOINT_SCALING_MODELS
    )
    is_checkpoint = (
        is_pythia_checkpoint_model_id
        if family == "pythia"
        else is_olmo2_checkpoint_model_id
    )

    values: dict[tuple[float, float], float] = {}
    params_values: set[float] = set()
    token_values: set[float] = set()
    for model_id, accuracy in latest[task].items():
        if not is_checkpoint(model_id) or base_model_id(model_id) not in bases:
            continue
        params_b = parse_params_billions(model_id)
        tokens_b = _checkpoint_tokens_b(model_id)
        if params_b is None or tokens_b is None:
            continue
        params_values.add(params_b)
        token_values.add(tokens_b)
        values[(params_b, tokens_b)] = accuracy

    params = sorted(params_values)
    tokens = sorted(token_values)
    matrix = [
        [values.get((params_b, tokens_b)) for tokens_b in tokens]
        for params_b in params
    ]
    return params, tokens, matrix


def plot(
    latest: dict[str, dict[str, float]],
    *,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required") from exc

    fig, axes = plt.subplots(
        len(_FAMILIES),
        len(_TASKS),
        figsize=(18, 9),
        squeeze=False,
        constrained_layout=True,
    )
    norm = Normalize(vmin=0, vmax=100)
    task_cmaps = {}
    for task in _TASKS:
        chance_position = _TASK_CHANCE[task] / 100
        task_cmaps[task] = LinearSegmentedColormap.from_list(
            f"{task}_chance",
            [
                (0.0, "#b2182b"),
                (chance_position, "#b2182b"),
                ((chance_position + 1.0) / 2.0, "#fee08b"),
                (1.0, "#1a9850"),
            ],
        )
        task_cmaps[task].set_bad("#e5e5e5")
    full_results = {
        family: full_models(latest, family=family) for family in _FAMILIES
    }

    for row, family in enumerate(_FAMILIES):
        for col, task in enumerate(_TASKS):
            ax = axes[row][col]
            params, tokens, sparse = build_matrix(
                latest, family=family, task=task
            )
            cmap = task_cmaps[task]
            matrix = np.array(
                [
                    [np.nan if value is None else value for value in values]
                    for values in sparse
                ],
                dtype=float,
            )
            ax.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="none",
            )

            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(
                [_format_tokens(value) for value in tokens],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels([_format_params(value) for value in params])
            ax.set_xlabel("Tokens seen")
            if col == 0:
                ax.set_ylabel(f"{_FAMILY_LABELS[family]}\nModel parameters")
            if row == 0:
                ax.set_title(_TASK_LABELS[task])

            for y, values in enumerate(sparse):
                for x, value in enumerate(values):
                    if value is None:
                        continue
                    red, green, blue, _ = cmap(norm(value))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    text_color = "white" if luminance < 0.5 else "black"
                    ax.text(
                        x,
                        y,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="medium",
                    )

            full_by_params = {
                parse_params_billions(model_id): accuracy
                for model_id, accuracy in full_results[family][task].items()
            }
            full_x = len(tokens) + 0.8
            for y, params_b in enumerate(params):
                value = full_by_params.get(params_b)
                if value is None:
                    continue
                facecolor = cmap(norm(value))
                ax.add_patch(
                    Rectangle(
                        (full_x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=facecolor,
                        edgecolor="white",
                        linewidth=1.5,
                    )
                )
                red, green, blue, _ = facecolor
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                ax.text(
                    full_x,
                    y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if luminance < 0.5 else "black",
                    fontsize=9,
                    fontweight="medium",
                )
            ax.set_xticks([*range(len(tokens)), full_x])
            ax.set_xticklabels(
                [
                    *[_format_tokens(value) for value in tokens],
                    _FULL_TOKEN_LABELS[family],
                ],
                rotation=45,
                ha="right",
            )
            ax.set_xlim(-0.5, full_x + 0.5)

            ax.set_xticks(
                [value - 0.5 for value in range(1, len(tokens))], minor=True
            )
            ax.set_yticks(
                [value - 0.5 for value in range(1, len(params))], minor=True
            )
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)

    for col, task in enumerate(_TASKS):
        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=task_cmaps[task]),
            ax=axes[:, col],
            location="bottom",
            shrink=0.82,
            pad=0.13,
            aspect=35,
        )
        colorbar.set_label(
            f"{_TASK_LABELS[task]} accuracy (%) — "
            f"chance {_TASK_CHANCE[task]:g}%"
        )
    fig.suptitle(title, fontsize=15)
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
        args.experiments,
        args.pythia_checkpoints,
        args.olmo_checkpoints,
    )
    output = args.output or (
        PLOTS_DIR / "checkpoint_parameter_token_accuracy_heatmaps.png"
    )
    title = args.title or (
        "Checkpoint accuracy by model parameters and tokens seen — "
        "forced choice, max_tasks=140"
    )
    plot(latest, output=output, title=title)


if __name__ == "__main__":
    main()
