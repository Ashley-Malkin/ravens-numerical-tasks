#!/usr/bin/env python3
"""Scaling with separate lines per n_examples (mixed run dates).

Supports Pythia and Qwen3 via ``--family``; instruction or completion via ``--prompt-type``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

_EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = _EVAL_DIR.parent

DatePrefixes = Union[str, Tuple[str, ...]]
NExampleDates = Dict[int, DatePrefixes]

INSTRUCTION_N_EXAMPLE_DATES: NExampleDates = {
    0: "2026-07-02",
    1: "2026-07-01",
    3: "2026-07-02",
}

# n=0 and n=1 on Jul 2/1; n=3 batch mostly Jul 2 (Qwen3 / 12B logged Jul 3).
COMPLETION_N_EXAMPLE_DATES: NExampleDates = {
    0: "2026-07-02",
    1: "2026-07-01",
    3: ("2026-07-02", "2026-07-03"),
}

_N_EXAMPLE_STYLE: dict[int, tuple[str, str, str]] = {
    0: ("o-", "tab:blue", "n=0 (2026-07-02)"),
    1: ("s-", "tab:orange", "n=1 (2026-07-01)"),
    3: ("^-", "tab:green", "n=3 (2026-07-02)"),
}

_COMPLETION_N3_LABEL = "n=3 (2026-07-02/03)"

_FAMILY_LABEL = {"pythia": "Pythia", "qwen3": "Qwen3"}

_BY_TASK_RE = re.compile(r"\*\*By task:\*\* (.+)$", re.MULTILINE)
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _header_matches_date(header: str, date_prefix: DatePrefixes) -> bool:
    if isinstance(date_prefix, tuple):
        return any(d in header for d in date_prefix)
    return date_prefix in header


def _match_icl_block(
    block: str,
    *,
    n_example_dates: NExampleDates,
    prompt_type: str,
    prompt_mode: str | None,
) -> int | None:
    header = block.split("\n", 1)[0]
    if f", {prompt_type})" not in header:
        return None
    if prompt_mode is not None and f"`--ravens-prompt-mode {prompt_mode}`" not in block:
        return None
    for n_examples, date_prefix in n_example_dates.items():
        if not _header_matches_date(header, date_prefix):
            continue
        if f"`--n-examples {n_examples}`" not in block:
            continue
        return n_examples
    return None


def _line_label(n_examples: int, prompt_type: str) -> str:
    _, _, label = _N_EXAMPLE_STYLE.get(n_examples, ("o-", "gray", f"n={n_examples}"))
    if prompt_type == "completion" and n_examples == 3:
        return _COMPLETION_N3_LABEL
    return label


def parse_scaling_by_n_examples(
    experiments_path: Path,
    *,
    family: str,
    n_example_dates: NExampleDates,
    prompt_type: str = "instruction",
    prompt_mode: str | None = "choice_only",
) -> dict[int, list[tuple[float, float]]]:
    """Return ``n_examples -> [(params_b, accuracy_pct), ...]`` for one model family."""
    _ensure_repo_on_path()
    from ravens_eval_models import model_family, parse_params_billions

    text = experiments_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)
    out: dict[int, list[tuple[float, float]]] = {n: [] for n in n_example_dates}

    for block in blocks:
        matched_n = _match_icl_block(
            block,
            n_example_dates=n_example_dates,
            prompt_type=prompt_type,
            prompt_mode=prompt_mode,
        )
        if matched_n is None:
            continue

        model_match = re.search(r"^### (.+)$", block, re.MULTILINE)
        overall_match = re.search(r"\*\*Overall:\*\* ([0-9.]+)%", block)
        if not model_match or not overall_match:
            continue

        model_id = model_match.group(1).strip()
        if model_family(model_id) != family:
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        out[matched_n].append((params_b, float(overall_match.group(1))))

    for n in out:
        out[n].sort(key=lambda x: x[0])
    return out


def parse_by_task_icl(
    experiments_path: Path,
    *,
    family: str,
    n_example_dates: NExampleDates,
    prompt_type: str = "instruction",
    prompt_mode: str | None = "choice_only",
) -> dict[str, dict[int, list[tuple[float, float]]]]:
    """Return ``task_type -> {n_examples -> [(params_b, accuracy_pct), ...]}``."""
    _ensure_repo_on_path()
    from experiment_log import TASK_TYPE_ORDER
    from ravens_eval_models import model_family, parse_params_billions

    text = experiments_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n---\n", text)
    out: dict[str, dict[int, list[tuple[float, float]]]] = {
        tt: {n: [] for n in n_example_dates} for tt in TASK_TYPE_ORDER
    }

    for block in blocks:
        matched_n = _match_icl_block(
            block,
            n_example_dates=n_example_dates,
            prompt_type=prompt_type,
            prompt_mode=prompt_mode,
        )
        if matched_n is None:
            continue

        model_match = re.search(r"^### (.+)$", block, re.MULTILINE)
        by_task_match = _BY_TASK_RE.search(block)
        if not model_match or not by_task_match:
            continue

        model_id = model_match.group(1).strip()
        if model_family(model_id) != family:
            continue
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue

        for task_type, acc_str in _TASK_ACC_RE.findall(by_task_match.group(1)):
            if task_type not in out:
                out[task_type] = {n: [] for n in n_example_dates}
            out[task_type][matched_n].append((params_b, float(acc_str)))

    for task_type in out:
        for n in n_example_dates:
            out[task_type][n].sort(key=lambda x: x[0])
    return out


def plot_icl_scaling(
    series: dict[int, list[tuple[float, float]]],
    *,
    output: Path,
    title: str,
    n_example_dates: NExampleDates,
    prompt_type: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_examples in sorted(series):
        points = series[n_examples]
        if not points:
            continue
        px, py = zip(*points)
        fmt, color, _ = _N_EXAMPLE_STYLE.get(n_examples, ("o-", "gray", f"n={n_examples}"))
        ax.plot(
            px,
            py,
            fmt,
            color=color,
            linewidth=2,
            markersize=8,
            label=_line_label(n_examples, prompt_type),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(title)
    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def plot_icl_facets(
    by_task: dict[str, dict[int, list[tuple[float, float]]]],
    *,
    output: Path,
    title: str,
    task_order: tuple[str, ...],
    n_example_dates: NExampleDates,
    prompt_type: str,
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
        series = by_task.get(task_type, {})

        for n_examples in sorted(n_example_dates):
            points = series.get(n_examples, [])
            if not points:
                continue
            fx, fy = zip(*points)
            fmt, color, _ = _N_EXAMPLE_STYLE.get(n_examples, ("o-", "gray", f"n={n_examples}"))
            ax.plot(
                fx,
                fy,
                fmt,
                color=color,
                linewidth=2,
                markersize=6,
                label=_line_label(n_examples, prompt_type),
            )

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
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle(title, fontsize=13, y=1.06)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def _config_for_prompt_type(prompt_type: str) -> tuple[NExampleDates, str | None, str]:
    if prompt_type == "completion":
        return COMPLETION_N_EXAMPLE_DATES, None, "completion"
    return INSTRUCTION_N_EXAMPLE_DATES, "choice_only", "instruction (choice-only)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling by n_examples (mixed dates); Pythia or Qwen3"
    )
    parser.add_argument(
        "--family",
        choices=("pythia", "qwen3"),
        default="qwen3",
    )
    parser.add_argument(
        "--prompt-type",
        choices=("instruction", "completion"),
        default="instruction",
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        default=_EVAL_DIR / "experiments.md",
    )
    parser.add_argument(
        "--prompt-mode",
        default=None,
        help="Override ravens-prompt-mode filter (default: choice_only for instruction)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--faceted",
        action="store_true",
        help="Per-task faceted plot (default: overall scaling only)",
    )
    args = parser.parse_args()

    n_example_dates, default_prompt_mode, prompt_label = _config_for_prompt_type(
        args.prompt_type
    )
    prompt_mode = (
        args.prompt_mode if args.prompt_mode is not None else default_prompt_mode
    )
    family_label = _FAMILY_LABEL[args.family]
    file_stem = f"{args.family}_{args.prompt_type}_icl"

    if args.faceted:
        _ensure_repo_on_path()
        from experiment_log import TASK_TYPE_ORDER

        by_task = parse_by_task_icl(
            args.experiments,
            family=args.family,
            n_example_dates=n_example_dates,
            prompt_type=args.prompt_type,
            prompt_mode=prompt_mode,
        )
        task_order = tuple(
            tt
            for tt in TASK_TYPE_ORDER
            if any(by_task.get(tt, {}).get(n) for n in n_example_dates)
        )
        if not task_order:
            raise SystemExit(
                f"No {family_label} {args.prompt_type} entries in {args.experiments}"
            )

        output = args.output or (_EVAL_DIR / f"{file_stem}_task_scaling.png")
        title = args.title or (
            f"{family_label} {prompt_label} — per-task accuracy by scale and n_examples "
            f"(20 tasks each)"
        )
        plot_icl_facets(
            by_task,
            output=output,
            title=title,
            task_order=task_order,
            n_example_dates=n_example_dates,
            prompt_type=args.prompt_type,
        )
        return

    series = parse_scaling_by_n_examples(
        args.experiments,
        family=args.family,
        n_example_dates=n_example_dates,
        prompt_type=args.prompt_type,
        prompt_mode=prompt_mode,
    )
    if not any(series.values()):
        raise SystemExit(
            f"No {family_label} {args.prompt_type} entries in {args.experiments}"
        )

    output = args.output or (_EVAL_DIR / f"{file_stem}_scaling.png")
    title = args.title or (
        f"{family_label} {prompt_label} — overall accuracy by scale and n_examples"
    )
    plot_icl_scaling(
        series,
        output=output,
        title=title,
        n_example_dates=n_example_dates,
        prompt_type=args.prompt_type,
    )


if __name__ == "__main__":
    main()
