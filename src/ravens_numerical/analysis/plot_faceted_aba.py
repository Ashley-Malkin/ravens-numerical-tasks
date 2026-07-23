#!/usr/bin/env python3
"""Faceted (per-model-family) ABA overall-accuracy scaling curves.

Same layout as ``faceted_icl_scaling.png`` (overall accuracy vs parameters, log
x-axis, shared axes, chance line) but for ``--task-type aba`` runs:

- Pythia / Qwen3: connected scaling lines for n=5, 10, and 15 (viridis colors).
- MiniBERTa: one dot per checkpoint at n=15 (45M for med-small 1M, 125M base).
- BabyLM: one dot per GPT-2 baseline at n=15 (~124M; tracks stack at x).
- Qwen3-8B-Base: star markers at x=8B (n=5/10/15 when present).
- Chance line at 50% (2-way forced choice).

Data: 2026-07-10 from ``experiments.md`` (Pythia / Qwen3 / MiniBERTa) and
``babyLMexperiments.md`` (BabyLM).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.models.registry import (
    is_babylm_model,
    is_miniberta_model,
    is_pythia_model,
    is_qwen3_base_model,
    is_qwen3_model,
    parse_params_billions,
)
from ravens_numerical.paths import BABYLM_EXPERIMENTS_MD, EXPERIMENTS_MD, PLOTS_DIR

_ABA_DATE = "2026-07-10"
_LINE_N_VALUES = (5, 10, 15)  # Pythia / Qwen3 overlay
_DOT_N_EXAMPLES = 15  # MiniBERTa / BabyLM
_CHANCE_PCT = 50.0

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")
_MAX_TASKS_RE = re.compile(r"\(max_tasks=(\d+)\)")

_FACETS = ("pythia", "qwen3", "miniberta", "babylm")
_FACET_LABEL = {
    "pythia": "Pythia",
    "qwen3": "Qwen3",
    "miniberta": "MiniBERTa",
    "babylm": "BabyLM",
}


def _n_example_colors(cmap_name: str = "viridis") -> dict[int, tuple]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    n = len(_LINE_N_VALUES)
    return {
        val: cmap(0.12 + 0.76 * (i / max(1, n - 1)))
        for i, val in enumerate(_LINE_N_VALUES)
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


def _is_aba_block(
    block: str,
    header: str,
    *,
    date_prefix: str,
    n_examples: int | None = None,
) -> bool:
    if date_prefix not in header or ", aba)" not in header:
        return False
    if "`--task-type aba`" not in block and ", aba)" not in header:
        return False
    if n_examples is not None and f"`--n-examples {n_examples}`" not in block:
        return False
    return True


def _max_tasks(header: str) -> int:
    m = _MAX_TASKS_RE.search(header)
    return int(m.group(1)) if m else 0


def parse_aba_family_lines(
    experiments_path: Path,
    *,
    family: str,
    n_examples: int,
    date_prefix: str = _ABA_DATE,
) -> list[tuple[float, float]]:
    """Return ``[(params_b, accuracy_pct), ...]`` for Pythia or Qwen3 instruct.

    Prefers the highest ``max_tasks`` when the same scale appears more than once
    (e.g. Qwen3-8B has both max_tasks=14 and 140). Excludes ``*-Base``.
    """
    by_params: dict[float, tuple[int, float]] = {}
    for block in _iter_blocks(experiments_path):
        header = _block_header(block)
        if not _is_aba_block(
            block, header, date_prefix=date_prefix, n_examples=n_examples
        ):
            continue
        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if family == "pythia":
            if not is_pythia_model(model_id):
                continue
        elif family == "qwen3":
            if not is_qwen3_model(model_id) or is_qwen3_base_model(model_id):
                continue
        else:
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        mt = _max_tasks(header)
        acc = float(overall_match.group(1))
        prev = by_params.get(params_b)
        if prev is None or mt >= prev[0]:
            by_params[params_b] = (mt, acc)
    return sorted((p, acc) for p, (_, acc) in by_params.items())


def parse_aba_qwen3_base(
    experiments_path: Path,
    *,
    n_examples: int,
    date_prefix: str = _ABA_DATE,
) -> tuple[float, float] | None:
    """Return ``(params_b, accuracy_pct)`` for Qwen3-8B-Base, if present."""
    best: tuple[int, float, float] | None = None  # max_tasks, params, acc
    for block in _iter_blocks(experiments_path):
        header = _block_header(block)
        if not _is_aba_block(
            block, header, date_prefix=date_prefix, n_examples=n_examples
        ):
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
        mt = _max_tasks(header)
        acc = float(overall_match.group(1))
        if best is None or mt >= best[0]:
            best = (mt, params_b, acc)
    if best is None:
        return None
    return best[1], best[2]


def parse_aba_dots(
    experiments_path: Path,
    *,
    family: str,
    n_examples: int = _DOT_N_EXAMPLES,
    date_prefix: str = _ABA_DATE,
) -> list[tuple[float, float, str]]:
    """One ``(params_b, accuracy_pct, model_id)`` per MiniBERTa or BabyLM checkpoint."""
    points: list[tuple[float, float, str]] = []
    for block in _iter_blocks(experiments_path):
        header = _block_header(block)
        if not _is_aba_block(
            block, header, date_prefix=date_prefix, n_examples=n_examples
        ):
            continue
        model_match = _MODEL_RE.search(block)
        overall_match = _OVERALL_RE.search(block)
        if not model_match or not overall_match:
            continue
        model_id = model_match.group(1).strip()
        if family == "miniberta":
            if not is_miniberta_model(model_id):
                continue
        elif family == "babylm":
            if not is_babylm_model(model_id):
                continue
        else:
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        points.append((params_b, float(overall_match.group(1)), model_id))
    points.sort(key=lambda x: (x[0], x[2]))
    return points


def _all_param_billions(
    *,
    pythia: dict[int, list[tuple[float, float]]],
    qwen3: dict[int, list[tuple[float, float]]],
    qwen3_base: dict[int, tuple[float, float]],
    miniberta: list[tuple[float, float, str]],
    babylm: list[tuple[float, float, str]],
) -> list[float]:
    values: list[float] = []
    for series in (pythia, qwen3):
        for points in series.values():
            values.extend(p[0] for p in points)
    values.extend(p[0] for p in miniberta)
    values.extend(p[0] for p in babylm)
    values.extend(pt[0] for pt in qwen3_base.values())
    return values


def plot_faceted_aba(
    *,
    pythia: dict[int, list[tuple[float, float]]],
    qwen3: dict[int, list[tuple[float, float]]],
    qwen3_base: dict[int, tuple[float, float]],
    miniberta: list[tuple[float, float, str]],
    babylm: list[tuple[float, float, str]],
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
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = [axes_grid[i // ncols][i % ncols] for i in range(len(_FACETS))]
    for idx in range(len(_FACETS), nrows * ncols):
        axes_grid[idx // ncols][idx % ncols].set_visible(False)

    colors = _n_example_colors()
    n15_color = colors[_DOT_N_EXAMPLES]

    line_series = {"pythia": pythia, "qwen3": qwen3}
    dot_series = {"miniberta": miniberta, "babylm": babylm}

    for idx, (ax, family) in enumerate(zip(axes, _FACETS)):
        if family in ("pythia", "qwen3"):
            for n_examples in _LINE_N_VALUES:
                points = line_series[family].get(n_examples, [])
                if not points:
                    continue
                px, py = zip(*points)
                ax.plot(
                    px,
                    py,
                    marker="o",
                    linestyle="-",
                    color=colors[n_examples],
                    linewidth=2,
                    markersize=8,
                )
            if family == "qwen3":
                for n_examples, (bx, by) in qwen3_base.items():
                    ax.plot(
                        bx,
                        by,
                        marker="*",
                        markersize=18,
                        color=colors.get(n_examples, n15_color),
                        linestyle="none",
                        zorder=4,
                    )
        else:
            dots = dot_series[family]
            if dots:
                ax.scatter(
                    [p[0] for p in dots],
                    [p[1] for p in dots],
                    marker="o",
                    color=n15_color,
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
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Parameters (billions, log scale)")
        if idx % ncols == 0:
            ax.set_ylabel("Overall accuracy (%)")

    params = _all_param_billions(
        pythia=pythia,
        qwen3=qwen3,
        qwen3_base=qwen3_base,
        miniberta=miniberta,
        babylm=babylm,
    )
    if params:
        xmin, xmax = min(params), max(params)
        for ax in axes:
            ax.set_xlim(xmin * 0.85, xmax * 1.12)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[n],
            marker="o",
            linewidth=2,
            label=f"n={n}",
        )
        for n in _LINE_N_VALUES
    ] + [
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="none",
            markersize=14,
            label="Qwen3-8B-Base",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=n15_color,
            linestyle="none",
            markersize=9,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"MiniBERTa / BabyLM (n={_DOT_N_EXAMPLES})",
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
        ncol=5,
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

    pythia = {
        n: parse_aba_family_lines(args.experiments, family="pythia", n_examples=n)
        for n in _LINE_N_VALUES
    }
    qwen3 = {
        n: parse_aba_family_lines(args.experiments, family="qwen3", n_examples=n)
        for n in _LINE_N_VALUES
    }
    qwen3_base: dict[int, tuple[float, float]] = {}
    for n in _LINE_N_VALUES:
        pt = parse_aba_qwen3_base(args.experiments, n_examples=n)
        if pt is not None:
            qwen3_base[n] = pt
    miniberta = parse_aba_dots(args.experiments, family="miniberta")
    babylm = parse_aba_dots(args.babylm_experiments, family="babylm")

    output = args.output or (PLOTS_DIR / "faceted_aba_scaling.png")
    title = args.title or (
        f"ABA overall accuracy by scale (n={','.join(map(str, _LINE_N_VALUES))}, "
        f"{_ABA_DATE})"
    )
    plot_faceted_aba(
        pythia=pythia,
        qwen3=qwen3,
        qwen3_base=qwen3_base,
        miniberta=miniberta,
        babylm=babylm,
        output=output,
        title=title,
    )


if __name__ == "__main__":
    main()
