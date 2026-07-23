#!/usr/bin/env python3
"""Faceted Ravens completion (forced choice) accuracy vs training tokens.

Four panels (Pythia checkpoints, OLMo checkpoints, BabyLM, MiniBERTa) in the
style of ``pythia_checkpoint_overall_combined_n1.png``: x-axis is tokens trained
(log scale, shared across facets), y-axis is overall accuracy.

Filters (latest log entry per model):

- ``max_tasks=140``
- ``--task-type ravens``
- ``--prompt-type completion``
- ``--score-mode forced_choice``
- ``--n-examples 1``

Full-size Pythia and OLMo 2 base models appear as stars on their panels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.models.registry import (
    BABYLM_SCALING_MODELS,
    MINIBERTA_SCALING_MODELS,
    OLMO2_BASE_MODELS,
    OLMO2_CHECKPOINT_SCALING_MODELS,
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    PYTHIA_FULL_TRAINING_TOKENS_B,
    base_model_id,
    is_babylm_model,
    is_miniberta_model,
    is_olmo2_base_model,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    is_pythia_model,
    olmo2_tokens_billions_from_revision,
    parse_checkpoint_model_id,
    parse_training_corpus_millions,
    pythia_step_from_revision,
    pythia_tokens_billions_at_step,
)
from ravens_numerical.paths import (
    BABYLM_EXPERIMENTS_MD,
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PLOTS_DIR,
    PYTHIA_CHECKPOINTS_MD,
)

_MAX_TASKS = "(max_tasks=140)"
_CHANCE_PCT = 25.0
_MIN_LOG_X_B = 0.0005
_EXCLUDED_OLMO_TOKEN_BUDGETS_B = {51.0}

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")

# Approximate full-training tokens (billions) for OLMo 2 base models.
_OLMO2_FULL_TRAINING_TOKENS_B: dict[str, float] = {
    "allenai/OLMo-2-0425-1B": 4000.0,
    "allenai/OLMo2-7B-1124": 3900.0,
    "allenai/OLMo-2-13B-1124": 3900.0,
}

_PYTHIA_LABEL = {
    "EleutherAI/pythia-70m-deduped": "Pythia 70M",
    "EleutherAI/pythia-160m-deduped": "Pythia 160M",
    "EleutherAI/pythia-1b-deduped": "Pythia 1B",
}
_PYTHIA_COLORS = {
    "EleutherAI/pythia-70m-deduped": "tab:blue",
    "EleutherAI/pythia-160m-deduped": "tab:orange",
    "EleutherAI/pythia-1b-deduped": "tab:green",
}
_OLMO_LABEL = {
    "allenai/OLMo-2-0425-1B": "OLMo 2 1B",
    "allenai/OLMo2-7B-1124": "OLMo 2 7B",
    "allenai/OLMo-2-13B-1124": "OLMo 2 13B",
}
_OLMO_COLORS = {
    "allenai/OLMo-2-0425-1B": "tab:blue",
    "allenai/OLMo2-7B-1124": "tab:orange",
    "allenai/OLMo-2-13B-1124": "tab:green",
}

_FACETS = ("pythia", "olmo2", "babylm", "miniberta")
_FACET_LABEL = {
    "pythia": "Pythia checkpoints",
    "olmo2": "OLMo 2 checkpoints",
    "babylm": "BabyLM",
    "miniberta": "MiniBERTa",
}


def _iter_blocks(experiments_path: Path):
    text = experiments_path.read_text(encoding="utf-8")
    parts = re.split(r"(?=^## \d{4}-\d{2}-\d{2} )", text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if part.startswith("## "):
            yield part


def _block_header(block: str) -> str:
    for line in block.splitlines():
        if line.strip():
            return line
    return ""


def _matches_forced_choice_ravens_completion(block: str, header: str) -> bool:
    if _MAX_TASKS not in header:
        return False
    if "forced_choice" not in block:
        return False
    if "prompt-type completion" not in block:
        return False
    if "task-type ravens" not in block and ", ravens)" not in header:
        return False
    if "`--n-examples 1`" not in block and "--n-examples 1" not in block:
        return False
    return True


def _parse_overall(block: str) -> float | None:
    match = _OVERALL_RE.search(block)
    return float(match.group(1)) if match else None


def _parse_model(block: str) -> str | None:
    match = _MODEL_RE.search(block)
    return match.group(1).strip() if match else None


def _latest_matching(
    *paths: Path,
) -> dict[str, float]:
    """``model_id -> overall accuracy %`` for the latest matching block."""
    latest: dict[str, float] = {}
    for path in paths:
        if not path.is_file():
            continue
        for block in _iter_blocks(path):
            header = _block_header(block)
            if not _matches_forced_choice_ravens_completion(block, header):
                continue
            model_id = _parse_model(block)
            acc = _parse_overall(block)
            if model_id is None or acc is None:
                continue
            latest[model_id] = acc
    return latest


def _pythia_tokens_b(model_id: str) -> float | None:
    _, revision = parse_checkpoint_model_id(model_id)
    if revision is None:
        return None
    step = pythia_step_from_revision(revision)
    if step is None:
        return None
    return pythia_tokens_billions_at_step(step)


def _olmo_tokens_b(model_id: str) -> float | None:
    _, revision = parse_checkpoint_model_id(model_id)
    if revision is None:
        return None
    tokens_b = olmo2_tokens_billions_from_revision(revision)
    if tokens_b is None:
        return None
    if tokens_b <= 0:
        return _MIN_LOG_X_B
    return tokens_b


def _corpus_tokens_b(model_id: str) -> float | None:
    millions = parse_training_corpus_millions(model_id)
    if millions is None:
        return None
    return millions / 1000.0


def parse_pythia_checkpoint_series(
    latest: dict[str, float],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, float]]:
    """Checkpoint lines and full-model stars for the Pythia panel."""
    checkpoints: dict[str, list[tuple[float, float]]] = {
        base: [] for base in PYTHIA_CHECKPOINT_SCALING_MODELS
    }
    finals: dict[str, float] = {}

    for model_id, acc in latest.items():
        if is_pythia_checkpoint_model_id(model_id):
            base = base_model_id(model_id)
            if base not in checkpoints:
                continue
            tokens_b = _pythia_tokens_b(model_id)
            if tokens_b is None:
                continue
            checkpoints[base].append((tokens_b, acc))
        elif is_pythia_model(model_id) and "@" not in model_id:
            base = base_model_id(model_id)
            if base in PYTHIA_CHECKPOINT_SCALING_MODELS:
                finals[base] = acc

    for base in checkpoints:
        checkpoints[base].sort(key=lambda x: x[0])
    return checkpoints, finals


def parse_olmo_checkpoint_series(
    latest: dict[str, float],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, float]]:
    """Checkpoint lines and full-model stars for the OLMo panel."""
    checkpoints: dict[str, list[tuple[float, float]]] = {
        base: [] for base in OLMO2_CHECKPOINT_SCALING_MODELS
    }
    finals: dict[str, float] = {}

    for model_id, acc in latest.items():
        if is_olmo2_checkpoint_model_id(model_id):
            base = base_model_id(model_id)
            if base not in checkpoints:
                continue
            tokens_b = _olmo_tokens_b(model_id)
            if tokens_b is None:
                continue
            if tokens_b in _EXCLUDED_OLMO_TOKEN_BUDGETS_B:
                continue
            checkpoints[base].append((tokens_b, acc))
        elif is_olmo2_base_model(model_id):
            finals[model_id] = acc

    for base in checkpoints:
        checkpoints[base].sort(key=lambda x: x[0])
    return checkpoints, finals


def parse_corpus_scatter(
    latest: dict[str, float],
    *,
    family: str,
) -> list[tuple[float, float]]:
    """``[(tokens_b, acc), ...]`` for BabyLM or MiniBERTa."""
    points: list[tuple[float, float]] = []
    for model_id, acc in latest.items():
        if family == "babylm":
            if not is_babylm_model(model_id):
                continue
            if model_id not in BABYLM_SCALING_MODELS:
                continue
        elif family == "miniberta":
            if not is_miniberta_model(model_id):
                continue
            if model_id not in MINIBERTA_SCALING_MODELS:
                continue
        else:
            raise ValueError(family)
        tokens_b = _corpus_tokens_b(model_id)
        if tokens_b is None:
            continue
        points.append((tokens_b, acc))
    points.sort(key=lambda x: (x[0], x[1]))
    return points


def _all_token_x_values(
    pythia_ckpt: dict[str, list[tuple[float, float]]],
    olmo_ckpt: dict[str, list[tuple[float, float]]],
    babylm: list[tuple[float, float]],
    miniberta: list[tuple[float, float]],
    *,
    include_full_stars: bool = True,
) -> list[float]:
    values: list[float] = []
    for series in (pythia_ckpt, olmo_ckpt):
        for points in series.values():
            values.extend(x for x, _ in points)
    for points in (babylm, miniberta):
        values.extend(x for x, _ in points)
    if include_full_stars:
        values.append(PYTHIA_FULL_TRAINING_TOKENS_B)
        values.extend(_OLMO2_FULL_TRAINING_TOKENS_B.values())
    return [v for v in values if v > 0]


def _plot_checkpoint_panel(
    ax,
    *,
    checkpoints: dict[str, list[tuple[float, float]]],
    finals: dict[str, float],
    labels: dict[str, str],
    colors: dict[str, str],
    full_tokens_b: dict[str, float] | float,
    title: str,
) -> list:
    handles = []
    for base, points in checkpoints.items():
        color = colors[base]
        label = labels[base]
        if points:
            tx, ty = zip(*points)
            (line,) = ax.plot(
                tx,
                ty,
                "o-",
                color=color,
                linewidth=2,
                markersize=7,
                label=label,
            )
            handles.append(line)
        elif base in labels:
            # Legend entry when only the full-model star is available.
            from matplotlib.lines import Line2D

            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    linewidth=2,
                    label=label,
                )
            )
        if base in finals:
            star_x = (
                full_tokens_b[base]
                if isinstance(full_tokens_b, dict)
                else full_tokens_b
            )
            ax.plot(
                star_x,
                finals[base],
                marker="*",
                markersize=14,
                color=color,
                linestyle="none",
                zorder=4,
            )
    ax.set_title(title)
    return handles


def plot_faceted(
    *,
    pythia_ckpt: dict[str, list[tuple[float, float]]],
    pythia_finals: dict[str, float],
    olmo_ckpt: dict[str, list[tuple[float, float]]],
    olmo_finals: dict[str, float],
    babylm: list[tuple[float, float]],
    miniberta: list[tuple[float, float]],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False, sharex=True, sharey=True)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    pythia_handles = _plot_checkpoint_panel(
        axes_flat[0],
        checkpoints=pythia_ckpt,
        finals=pythia_finals,
        labels=_PYTHIA_LABEL,
        colors=_PYTHIA_COLORS,
        full_tokens_b=PYTHIA_FULL_TRAINING_TOKENS_B,
        title=_FACET_LABEL["pythia"],
    )
    olmo_handles = _plot_checkpoint_panel(
        axes_flat[1],
        checkpoints=olmo_ckpt,
        finals=olmo_finals,
        labels=_OLMO_LABEL,
        colors=_OLMO_COLORS,
        full_tokens_b=_OLMO2_FULL_TRAINING_TOKENS_B,
        title=_FACET_LABEL["olmo2"],
    )

    scatter_color = plt.get_cmap("viridis")(0.55)
    for ax, points, facet in zip(
        (axes_flat[2], axes_flat[3]),
        (babylm, miniberta),
        ("babylm", "miniberta"),
    ):
        if points:
            ax.scatter(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                color=scatter_color,
                s=80 if facet == "babylm" else 60,
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
            )
        ax.set_title(_FACET_LABEL[facet])

    token_x = _all_token_x_values(
        pythia_ckpt, olmo_ckpt, babylm, miniberta, include_full_stars=True
    )
    if token_x:
        xmin, xmax = min(token_x), max(token_x)
        for ax in axes_flat:
            ax.set_xlim(max(_MIN_LOG_X_B, xmin * 0.7), xmax * 1.15)

    for idx, ax in enumerate(axes_flat):
        ax.set_xscale("log")
        ax.axhline(_CHANCE_PCT, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Tokens trained (billions, log scale)")
        if idx % 2 == 0:
            ax.set_ylabel("Overall accuracy (%)")

    extra_handles = [
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
            label=f"Chance ({_CHANCE_PCT:g}%)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=scatter_color,
            linestyle="none",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="BabyLM / MiniBERTa",
        ),
    ]
    fig.legend(
        handles=pythia_handles + olmo_handles + extra_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.14, hspace=0.28, wspace=0.22)
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

    latest = _latest_matching(
        args.experiments,
        args.babylm_experiments,
        args.pythia_checkpoints,
        args.olmo_checkpoints,
    )
    if not latest:
        raise SystemExit("No matching forced-choice Ravens completion data found")

    pythia_ckpt, pythia_finals = parse_pythia_checkpoint_series(latest)
    olmo_ckpt, olmo_finals = parse_olmo_checkpoint_series(latest)
    babylm = parse_corpus_scatter(latest, family="babylm")
    miniberta = parse_corpus_scatter(latest, family="miniberta")

    if not any(pythia_ckpt.values()) and not pythia_finals:
        raise SystemExit("No Pythia checkpoint or final data found")
    if not any(olmo_ckpt.values()) and not olmo_finals:
        raise SystemExit("No OLMo checkpoint or final data found")

    output = args.output or (PLOTS_DIR / "faceted_checkpoints_forced_choice_scaling.png")
    title = args.title or (
        "Ravens completion (forced choice) vs tokens trained — max_tasks=140"
    )
    plot_faceted(
        pythia_ckpt=pythia_ckpt,
        pythia_finals=pythia_finals,
        olmo_ckpt=olmo_ckpt,
        olmo_finals=olmo_finals,
        babylm=babylm,
        miniberta=miniberta,
        output=output,
        title=title,
    )


if __name__ == "__main__":
    main()
