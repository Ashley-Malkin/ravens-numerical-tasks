#!/usr/bin/env python3
"""Faceted Ravens completion accuracy (forced choice) vs model scale.

Uses the most recent log entry per model matching:

- ``max_tasks=140``
- ``--task-type ravens``
- ``--prompt-type completion``
- ``--score-mode forced_choice``

Layout matches ``faceted_icl_scaling.png`` (3 panels top, 2 centered bottom;
shared log-x across facets):

- Pythia / Qwen3: connected lines (final checkpoints only).
- OLMo 2: base (star) vs instruct (circle) lines.
- MiniBERTa / BabyLM: one dot per checkpoint/baseline.
- Qwen3-8B-Base: extra star at x=8B.
- Chance line at 25%.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.models.registry import (
    is_babylm_model,
    is_miniberta_model,
    is_olmo2_base_model,
    is_olmo2_instruct_model,
    is_olmo2_model,
    is_pythia_model,
    is_qwen3_base_model,
    is_qwen3_model,
    parse_params_billions,
)
from ravens_numerical.paths import BABYLM_EXPERIMENTS_MD, EXPERIMENTS_MD, PLOTS_DIR

_MAX_TASKS = "(max_tasks=140)"
_CHANCE_PCT = 25.0

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")

_FACETS = ("pythia", "qwen3", "olmo2", "miniberta", "babylm")
_FACET_LABEL = {
    "pythia": "Pythia",
    "qwen3": "Qwen3",
    "olmo2": "OLMo 2",
    "miniberta": "MiniBERTa",
    "babylm": "BabyLM",
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
    return True


def parse_forced_choice_ravens(
    *paths: Path,
) -> dict[str, list[tuple[float, float, str]]]:
    """Return ``{family -> [(params_b, acc_pct, model_id), ...]}`` (latest per model)."""
    latest: dict[str, tuple[float, float, str]] = {}

    for path in paths:
        if not path.is_file():
            continue
        for block in _iter_blocks(path):
            header = _block_header(block)
            if not _matches_forced_choice_ravens_completion(block, header):
                continue
            model_match = _MODEL_RE.search(block)
            overall_match = _OVERALL_RE.search(block)
            if not model_match or not overall_match:
                continue
            model_id = model_match.group(1).strip()
            params_b = parse_params_billions(model_id)
            if params_b is None:
                continue
            acc = float(overall_match.group(1))
            latest[model_id] = (params_b, acc, model_id)

    by_family: dict[str, list[tuple[float, float, str]]] = {
        f: [] for f in _FACETS
    }
    qwen3_base: list[tuple[float, float, str]] = []

    for model_id, (params_b, acc, _) in latest.items():
        if is_pythia_model(model_id) and "@" not in model_id:
            by_family["pythia"].append((params_b, acc, model_id))
        elif is_qwen3_base_model(model_id):
            qwen3_base.append((params_b, acc, model_id))
        elif is_qwen3_model(model_id):
            by_family["qwen3"].append((params_b, acc, model_id))
        elif is_olmo2_model(model_id) and "@" not in model_id:
            by_family["olmo2"].append((params_b, acc, model_id))
        elif is_miniberta_model(model_id):
            by_family["miniberta"].append((params_b, acc, model_id))
        elif is_babylm_model(model_id):
            by_family["babylm"].append((params_b, acc, model_id))

    for family in by_family:
        by_family[family].sort(key=lambda x: (x[0], x[2]))

    # Attach qwen3_base on the return object via a sentinel key handled in plot.
    by_family["_qwen3_base"] = qwen3_base
    return by_family


def _split_olmo2(
    points: list[tuple[float, float, str]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    base: list[tuple[float, float]] = []
    instruct: list[tuple[float, float]] = []
    for params_b, acc, model_id in points:
        if is_olmo2_base_model(model_id):
            base.append((params_b, acc))
        elif is_olmo2_instruct_model(model_id):
            instruct.append((params_b, acc))
    base.sort(key=lambda x: x[0])
    instruct.sort(key=lambda x: x[0])
    return base, instruct


def _all_param_billions(
    by_family: dict[str, list[tuple[float, float, str]]],
) -> list[float]:
    values: list[float] = []
    for family in _FACETS:
        values.extend(p[0] for p in by_family.get(family, []))
    values.extend(p[0] for p in by_family.get("_qwen3_base", []))
    return values


def plot_faceted(
    by_family: dict[str, list[tuple[float, float, str]]],
    *,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    top_n, bottom_n = 3, 2
    assert len(_FACETS) == top_n + bottom_n
    panel_span = bottom_n
    grid_cols = top_n * panel_span
    fig = plt.figure(figsize=(5.5 * top_n, 4.8 * 2))
    gs = GridSpec(2, grid_cols, figure=fig, wspace=0.35, hspace=0.35)
    axes: list = []
    for i in range(top_n):
        axes.append(fig.add_subplot(gs[0, i * panel_span : (i + 1) * panel_span]))
    bottom_offset = (grid_cols - bottom_n * panel_span) // 2
    for i in range(bottom_n):
        start = bottom_offset + i * panel_span
        axes.append(
            fig.add_subplot(
                gs[1, start : start + panel_span],
                sharex=axes[0],
                sharey=axes[0],
            )
        )
    for ax in axes[1:top_n]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])

    color = plt.get_cmap("viridis")(0.55)
    qwen3_base = by_family.get("_qwen3_base", [])

    for idx, (ax, family) in enumerate(zip(axes, _FACETS)):
        points = by_family.get(family, [])
        if family in ("pythia", "qwen3") and points:
            px, py = zip(*[(p[0], p[1]) for p in points])
            ax.plot(px, py, marker="o", linestyle="-", color=color, linewidth=2, markersize=8)
            if family == "qwen3" and qwen3_base:
                bx, by = qwen3_base[0][0], qwen3_base[0][1]
                ax.plot(
                    bx,
                    by,
                    marker="*",
                    markersize=18,
                    color=color,
                    linestyle="none",
                    zorder=4,
                )
        elif family == "olmo2" and points:
            base, instruct = _split_olmo2(points)
            for series, marker in ((base, "*"), (instruct, "o")):
                if not series:
                    continue
                px, py = zip(*series)
                ax.plot(
                    px,
                    py,
                    marker=marker,
                    linestyle="-",
                    color=color,
                    linewidth=2,
                    markersize=10 if marker == "*" else 8,
                )
        elif family in ("miniberta", "babylm") and points:
            ax.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                color=color,
                s=90 if family == "babylm" else 70,
                zorder=3,
                edgecolors="black",
                linewidths=0.5,
            )

        ax.set_xscale("log")
        ax.set_title(_FACET_LABEL[family])
        ax.axhline(_CHANCE_PCT, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx >= top_n:
            ax.set_xlabel("Parameters (billions, log scale)")
        if idx in (0, top_n):
            ax.set_ylabel("Overall accuracy (%)")

    params = _all_param_billions(by_family)
    if params:
        xmin, xmax = min(params), max(params)
        for ax in axes:
            ax.set_xlim(xmin * 0.85, xmax * 1.12)

    handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=2, label="Forced choice (n=1)"),
        Line2D(
            [0],
            [0],
            marker="*",
            color=color,
            linestyle="none",
            markersize=14,
            label="Qwen3-8B-Base / OLMo 2 base",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            linestyle="none",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="MiniBERTa / BabyLM (per checkpoint)",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Chance ({_CHANCE_PCT:g}%)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument(
        "--babylm-experiments", type=Path, default=BABYLM_EXPERIMENTS_MD
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    by_family = parse_forced_choice_ravens(args.experiments, args.babylm_experiments)
    output = args.output or (PLOTS_DIR / "faceted_forced_choice_scaling.png")
    title = args.title or (
        "Ravens completion (forced choice) accuracy by scale — max_tasks=140"
    )
    plot_faceted(by_family, output=output, title=title)


if __name__ == "__main__":
    main()
