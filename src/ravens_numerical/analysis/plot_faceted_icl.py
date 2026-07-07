#!/usr/bin/env python3
"""Faceted (per-model-family) overall-accuracy scaling curves.

Follows the format of ``pythia_completion_icl_scaling.png`` (overall accuracy vs
parameters, log x-axis, colored by ``n_examples``) but facets by model family and
overlays completion (solid) and instruction (dashed) runs in one figure:

- Pythia / Qwen3: completion n=0/1/3 (solid) + instruction n=0/1/3 (dashed).
- MiniBERTa: completion n=1 only, one dot per checkpoint (45M for the 1M
  med-small models, 125M for the base 10M/100M/1B models).
- BabyLM: completion (filled) + instruction (open) n=1 dots; both GPT-2
  baselines are ~124M params (they differ in training-corpus track), so they
  stack at the same x.
- Qwen3-8B-Base: instruction + completion (n=1) as extra dots at x=8B.
- Chance line at 25%.

The n_examples run dates match ``plot_qwen3_instruction_icl`` (n=0 Jul 2, n=1
Jul 1, n=3 Jul 2/03); MiniBERTa and Qwen3-8B-Base come from Jul 6; BabyLM
comes from Jul 7.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.plot_qwen3_instruction_icl import (
    COMPLETION_N_EXAMPLE_DATES,
    INSTRUCTION_N_EXAMPLE_DATES,
    _N_EXAMPLE_STYLE,
    parse_scaling_by_n_examples,
)
from ravens_numerical.models.registry import (
    is_babylm_model,
    is_miniberta_model,
    is_qwen3_base_model,
    parse_params_billions,
)
from ravens_numerical.paths import BABYLM_EXPERIMENTS_MD, EXPERIMENTS_MD, PLOTS_DIR

_MINIBERTA_DATE = "2026-07-06"
_QWEN3_BASE_DATE = "2026-07-06"
_BABYLM_DATE = "2026-07-07"
_MINIBERTA_MAX_TASKS = "(max_tasks=140)"
_BABYLM_MAX_TASKS = "(max_tasks=140)"

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")

_FACETS = ("pythia", "qwen3", "miniberta", "babylm")
_FACET_LABEL = {
    "pythia": "Pythia",
    "qwen3": "Qwen3",
    "miniberta": "MiniBERTa",
    "babylm": "BabyLM",
}

# n_examples ordered low→high; colored with a sequential palette so more
# in-context examples map to progressively brighter colors.
_N_VALUES = (0, 1, 3)


def _n_example_colors(cmap_name: str = "viridis") -> dict[int, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    # Sample within [0.12, 0.88] to avoid the darkest/lightest extremes.
    n = len(_N_VALUES)
    return {
        val: cmap(0.12 + 0.76 * (i / max(1, n - 1)))
        for i, val in enumerate(_N_VALUES)
    }


def _iter_blocks(experiments_path: Path):
    text = experiments_path.read_text(encoding="utf-8")
    return re.split(r"\n---\n", text)


def parse_miniberta_completion(
    experiments_path: Path,
    *,
    date_prefix: str = _MINIBERTA_DATE,
    n_examples: int = 1,
) -> list[tuple[float, float, str]]:
    """One ``(params_b, accuracy_pct, model_id)`` per MiniBERTa completion checkpoint."""
    points: list[tuple[float, float, str]] = []
    for block in _iter_blocks(experiments_path):
        header = block.split("\n", 1)[0]
        if date_prefix not in header or ", completion)" not in header:
            continue
        if _MINIBERTA_MAX_TASKS not in header:
            continue
        if f"`--n-examples {n_examples}`" not in block:
            continue
        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if not is_miniberta_model(model_id):
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        points.append((params_b, float(overall_match.group(1)), model_id))
    points.sort(key=lambda x: (x[0], x[2]))
    return points


def parse_qwen3_base(
    experiments_path: Path,
    *,
    date_prefix: str = _QWEN3_BASE_DATE,
    n_examples: int = 1,
) -> dict[str, tuple[float, float]]:
    """Return ``{prompt_type -> (params_b, accuracy_pct)}`` for Qwen3-8B-Base."""
    out: dict[str, tuple[float, float]] = {}
    for block in _iter_blocks(experiments_path):
        header = block.split("\n", 1)[0]
        if date_prefix not in header:
            continue
        if f"`--n-examples {n_examples}`" not in block:
            continue
        prompt_type = None
        if ", instruction)" in header:
            prompt_type = "instruction"
        elif ", completion)" in header:
            prompt_type = "completion"
        if prompt_type is None:
            continue
        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if not is_qwen3_base_model(model_id):
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        out[prompt_type] = (params_b, float(overall_match.group(1)))
    return out


def parse_babylm(
    experiments_path: Path,
    *,
    date_prefix: str = _BABYLM_DATE,
    n_examples: int = 1,
) -> dict[str, list[tuple[float, float, str]]]:
    """Return ``{prompt_type -> [(params_b, accuracy_pct, model_id), ...]}`` for BabyLM."""
    out: dict[str, list[tuple[float, float, str]]] = {}
    for block in _iter_blocks(experiments_path):
        header = block.split("\n", 1)[0]
        if date_prefix not in header or _BABYLM_MAX_TASKS not in header:
            continue
        if f"`--n-examples {n_examples}`" not in block:
            continue
        prompt_type = None
        if ", instruction)" in header:
            prompt_type = "instruction"
        elif ", completion)" in header:
            prompt_type = "completion"
        if prompt_type is None:
            continue
        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if not is_babylm_model(model_id):
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        out.setdefault(prompt_type, []).append(
            (params_b, float(overall_match.group(1)), model_id)
        )
    for prompt_type in out:
        out[prompt_type].sort(key=lambda x: (x[0], x[2]))
    return out


def _plot_family_lines(ax, series, *, linestyle: str, colors: dict[int, tuple]) -> None:
    for n_examples in sorted(series):
        points = series[n_examples]
        if not points:
            continue
        px, py = zip(*points)
        marker, _, _ = _N_EXAMPLE_STYLE.get(
            n_examples, ("o-", "gray", f"n={n_examples}")
        )
        marker = marker.rstrip("-")
        ax.plot(
            px,
            py,
            marker=marker,
            linestyle=linestyle,
            color=colors.get(n_examples, "gray"),
            linewidth=2,
            markersize=8,
        )


def plot_faceted(
    *,
    completion: dict[str, dict[int, list[tuple[float, float]]]],
    instruction: dict[str, dict[int, list[tuple[float, float]]]],
    miniberta: list[tuple[float, float, str]],
    qwen3_base: dict[str, tuple[float, float]],
    babylm: dict[str, list[tuple[float, float, str]]],
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
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False, sharey=True
    )
    axes = [axes_grid[i // ncols][i % ncols] for i in range(len(_FACETS))]

    colors = _n_example_colors()
    n1_color = colors[1]

    for idx, (ax, family) in enumerate(zip(axes, _FACETS)):
        if family == "miniberta":
            if miniberta:
                mx = [p[0] for p in miniberta]
                my = [p[1] for p in miniberta]
                ax.scatter(
                    mx,
                    my,
                    marker="o",
                    color=n1_color,
                    s=70,
                    zorder=3,
                    edgecolors="black",
                    linewidths=0.5,
                )
        elif family == "babylm":
            comp = babylm.get("completion", [])
            if comp:
                ax.scatter(
                    [p[0] for p in comp],
                    [p[1] for p in comp],
                    marker="o",
                    color=n1_color,
                    s=90,
                    zorder=3,
                    edgecolors="black",
                    linewidths=0.5,
                )
            instr = babylm.get("instruction", [])
            if instr:
                ax.scatter(
                    [p[0] for p in instr],
                    [p[1] for p in instr],
                    marker="o",
                    facecolors="white",
                    edgecolors=n1_color,
                    linewidths=1.5,
                    s=90,
                    zorder=4,
                )
        else:
            _plot_family_lines(ax, completion.get(family, {}), linestyle="-", colors=colors)
            _plot_family_lines(ax, instruction.get(family, {}), linestyle="--", colors=colors)

        if family == "qwen3" and qwen3_base:
            if "completion" in qwen3_base:
                bx, by = qwen3_base["completion"]
                ax.plot(
                    bx, by, marker="*", markersize=18, color=n1_color,
                    linestyle="none", zorder=4,
                )
            if "instruction" in qwen3_base:
                bx, by = qwen3_base["instruction"]
                ax.plot(
                    bx, by, marker="*", markersize=18, markerfacecolor="white",
                    markeredgecolor=n1_color, markeredgewidth=1.5,
                    linestyle="none", zorder=4,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_title(_FACET_LABEL[family])
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("Overall accuracy (%)")

    color_handles = [
        Line2D([0], [0], color=colors[n], marker="o", linewidth=2, label=f"n={n}")
        for n in _N_VALUES
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Completion"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Instruction"),
    ]
    extra_handles = [
        Line2D([0], [0], marker="*", color=n1_color, linestyle="none", markersize=14,
               label="Qwen3-8B-Base completion (n=1)"),
        Line2D([0], [0], marker="*", markerfacecolor="white", markeredgecolor=n1_color,
               markeredgewidth=1.5, linestyle="none", markersize=14,
               label="Qwen3-8B-Base instruction (n=1)"),
        Line2D([0], [0], marker="o", color=n1_color, linestyle="none", markersize=9,
               markeredgecolor="black", markeredgewidth=0.5,
               label="MiniBERTa completion (n=1, per checkpoint)"),
        Line2D([0], [0], marker="o", color=n1_color, linestyle="none", markersize=9,
               markeredgecolor="black", markeredgewidth=0.5,
               label="BabyLM completion (n=1, per baseline)"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=n1_color,
               markeredgewidth=1.5, linestyle="none", markersize=9,
               label="BabyLM instruction (n=1, per baseline)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Chance (25%)"),
    ]

    fig.legend(
        handles=color_handles + style_handles + extra_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
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

    completion: dict[str, dict[int, list[tuple[float, float]]]] = {}
    instruction: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for family in ("pythia", "qwen3"):
        completion[family] = parse_scaling_by_n_examples(
            args.experiments,
            family=family,
            n_example_dates=COMPLETION_N_EXAMPLE_DATES,
            prompt_type="completion",
            prompt_mode=None,
        )
        instruction[family] = parse_scaling_by_n_examples(
            args.experiments,
            family=family,
            n_example_dates=INSTRUCTION_N_EXAMPLE_DATES,
            prompt_type="instruction",
            prompt_mode="choice_only",
        )

    miniberta = parse_miniberta_completion(args.experiments)
    qwen3_base = parse_qwen3_base(args.experiments)
    babylm = parse_babylm(args.babylm_experiments)

    output = args.output or (PLOTS_DIR / "faceted_icl_scaling.png")
    title = args.title or (
        "Overall accuracy by scale — completion (solid) vs instruction (dashed)"
    )
    plot_faceted(
        completion=completion,
        instruction=instruction,
        miniberta=miniberta,
        qwen3_base=qwen3_base,
        babylm=babylm,
        output=output,
        title=title,
    )


if __name__ == "__main__":
    main()
