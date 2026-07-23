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
- OLMo 2: completion n=1 from 2026-07-08 and n=0 from 2026-07-09;
  instruction from 2026-07-09; base checkpoints use star markers,
  instruct-tuned use circle markers.
- Qwen3-8B-Base: instruction + completion (n=1) as extra dots at x=8B.
- Chance line at 25%.

The n_examples run dates match ``plot_qwen3_instruction_icl`` (n=0 Jul 2, n=1
Jul 1, n=3 Jul 2/03); MiniBERTa and Qwen3-8B-Base come from Jul 6; BabyLM
comes from Jul 7; OLMo 2 completion from Jul 8/9 and instruction from Jul 9.
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
    is_olmo2_base_model,
    is_olmo2_instruct_model,
    is_olmo2_model,
    is_qwen3_base_model,
    parse_params_billions,
)
from ravens_numerical.paths import BABYLM_EXPERIMENTS_MD, EXPERIMENTS_MD, PLOTS_DIR

_MINIBERTA_DATE = "2026-07-06"
_QWEN3_BASE_DATE = "2026-07-06"
_BABYLM_DATE = "2026-07-07"
_OLMO2_COMPLETION_DATES = ("2026-07-08", "2026-07-09")  # n=1 Jul 8; n=0 Jul 9
_OLMO2_INSTRUCTION_DATES = ("2026-07-09",)
_MINIBERTA_MAX_TASKS = "(max_tasks=140)"
_BABYLM_MAX_TASKS = "(max_tasks=140)"
_OLMO2_MAX_TASKS = "(max_tasks=140)"

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
    """Yield experiment log blocks.

    Prefer ``---`` separators; also split on ``## YYYY-MM-DD`` headers so a
    missing ``---`` does not glue two runs into one block.
    """
    text = experiments_path.read_text(encoding="utf-8")
    # Split before each dated experiment heading (keeps the heading in the block).
    parts = re.split(r"(?=^## \d{4}-\d{2}-\d{2} )", text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if part.startswith("## "):
            yield part


def _block_header(block: str) -> str:
    """First non-empty line of a log block (handles a blank line after ``---``)."""
    for line in block.splitlines():
        if line.strip():
            return line
    return ""


def parse_miniberta_completion(
    experiments_path: Path,
    *,
    date_prefix: str = _MINIBERTA_DATE,
    n_examples: int = 1,
) -> list[tuple[float, float, str]]:
    """One ``(params_b, accuracy_pct, model_id)`` per MiniBERTa completion checkpoint."""
    points: list[tuple[float, float, str]] = []
    for block in _iter_blocks(experiments_path):
        header = _block_header(block)
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
        header = _block_header(block)
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
        header = _block_header(block)
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


def parse_olmo2(
    experiments_path: Path,
    *,
    completion_dates: tuple[str, ...] = _OLMO2_COMPLETION_DATES,
    instruction_dates: tuple[str, ...] = _OLMO2_INSTRUCTION_DATES,
) -> dict[str, dict[str, dict[int, list[tuple[float, float]]]]]:
    """Return ``{prompt_type -> {base|instruct -> {n -> [(params_b, acc), ...]}}}``.

    Later blocks for the same ``(prompt_type, variant, n, params_b)`` overwrite
    earlier ones so the most recent run wins.
    """
    out: dict[str, dict[str, dict[int, list[tuple[float, float]]]]] = {
        "completion": {"base": {}, "instruct": {}},
        "instruction": {"base": {}, "instruct": {}},
    }

    for block in _iter_blocks(experiments_path):
        header = _block_header(block)
        if _OLMO2_MAX_TASKS not in header:
            continue

        prompt_type = None
        if ", completion)" in header and any(d in header for d in completion_dates):
            prompt_type = "completion"
        elif ", instruction)" in header and any(d in header for d in instruction_dates):
            prompt_type = "instruction"
        if prompt_type is None:
            continue

        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue

        model_id = model_match.group(1).strip()
        if not is_olmo2_model(model_id):
            continue
        if is_olmo2_base_model(model_id):
            variant = "base"
        elif is_olmo2_instruct_model(model_id):
            variant = "instruct"
        else:
            continue

        n_match = re.search(r"`--n-examples (\d+)`", block)
        if not n_match:
            continue
        n_examples = int(n_match.group(1))

        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue

        bucket = out[prompt_type][variant]
        points = bucket.setdefault(n_examples, [])
        acc = float(overall_match.group(1))
        # Replace any earlier point at the same scale (most recent run wins).
        points[:] = [p for p in points if p[0] != params_b]
        points.append((params_b, acc))

    for prompt_type in out:
        for variant in out[prompt_type]:
            for n in out[prompt_type][variant]:
                out[prompt_type][variant][n].sort(key=lambda x: x[0])
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


def _plot_olmo2_lines(
    ax,
    series: dict[int, list[tuple[float, float]]],
    *,
    linestyle: str,
    colors: dict[int, tuple],
    marker: str,
) -> None:
    for n_examples in sorted(series):
        points = series[n_examples]
        if not points:
            continue
        px, py = zip(*points)
        ax.plot(
            px,
            py,
            marker=marker,
            linestyle=linestyle,
            color=colors.get(n_examples, "gray"),
            linewidth=2,
            markersize=10 if marker == "*" else 8,
        )


def _all_param_billions(
    *,
    completion: dict[str, dict[int, list[tuple[float, float]]]],
    instruction: dict[str, dict[int, list[tuple[float, float]]]],
    miniberta: list[tuple[float, float, str]],
    qwen3_base: dict[str, tuple[float, float]],
    babylm: dict[str, list[tuple[float, float, str]]],
    olmo2: dict[str, dict[str, dict[int, list[tuple[float, float]]]]],
) -> list[float]:
    values: list[float] = []
    for family in ("pythia", "qwen3"):
        for series in (completion.get(family, {}), instruction.get(family, {})):
            for points in series.values():
                values.extend(p[0] for p in points)
    values.extend(p[0] for p in miniberta)
    for series in babylm.values():
        values.extend(p[0] for p in series)
    for pt in qwen3_base.values():
        values.append(pt[0])
    for prompt_type in olmo2.values():
        for variant in prompt_type.values():
            for points in variant.values():
                values.extend(p[0] for p in points)
    return values


def plot_faceted(
    *,
    completion: dict[str, dict[int, list[tuple[float, float]]]],
    instruction: dict[str, dict[int, list[tuple[float, float]]]],
    miniberta: list[tuple[float, float, str]],
    qwen3_base: dict[str, tuple[float, float]],
    babylm: dict[str, list[tuple[float, float, str]]],
    olmo2: dict[str, dict[str, dict[int, list[tuple[float, float]]]]],
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    # 3 panels on top, 2 centered on the bottom (5 facets total).
    top_n, bottom_n = 3, 2
    assert len(_FACETS) == top_n + bottom_n
    panel_span = bottom_n  # each panel spans this many grid columns
    grid_cols = top_n * panel_span
    fig = plt.figure(figsize=(5.5 * top_n, 4.8 * 2))
    gs = GridSpec(2, grid_cols, figure=fig, wspace=0.35, hspace=0.35)
    axes: list = []
    for i in range(top_n):
        axes.append(
            fig.add_subplot(gs[0, i * panel_span : (i + 1) * panel_span])
        )
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
        elif family == "olmo2":
            for variant, marker in (("base", "*"), ("instruct", "o")):
                _plot_olmo2_lines(
                    ax,
                    olmo2.get("completion", {}).get(variant, {}),
                    linestyle="-",
                    colors=colors,
                    marker=marker,
                )
                _plot_olmo2_lines(
                    ax,
                    olmo2.get("instruction", {}).get(variant, {}),
                    linestyle="--",
                    colors=colors,
                    marker=marker,
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
        ax.set_title(_FACET_LABEL[family])
        ax.axhline(25.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        if idx >= top_n:
            ax.set_xlabel("Parameters (billions, log scale)")
        if idx in (0, top_n):
            ax.set_ylabel("Overall accuracy (%)")

    params = _all_param_billions(
        completion=completion,
        instruction=instruction,
        miniberta=miniberta,
        qwen3_base=qwen3_base,
        babylm=babylm,
        olmo2=olmo2,
    )
    if params:
        xmin, xmax = min(params), max(params)
        for ax in axes:
            ax.set_xlim(xmin * 0.85, xmax * 1.12)

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
        Line2D([0], [0], marker="*", color="black", linestyle="-", linewidth=2,
               markersize=12, label="OLMo 2 base"),
        Line2D([0], [0], marker="o", color="black", linestyle="-", linewidth=2,
               markersize=8, label="OLMo 2 instruct"),
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
    olmo2 = parse_olmo2(args.experiments)

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
        olmo2=olmo2,
        output=output,
        title=title,
    )


if __name__ == "__main__":
    main()
