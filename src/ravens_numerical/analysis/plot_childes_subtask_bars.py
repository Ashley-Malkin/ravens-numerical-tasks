#!/usr/bin/env python3
"""CHILDES ladder subtask bar charts: base vs finetuned by word budget.

One facet per Easy Ravens subtask. Within each facet, x = pretraining
word budget (1M / 5M / 12M / 24M) with paired bars for base vs SFT.

Sources:

- Base: ``artifacts/logs/childesExperiments.md``
- Finetuned: ``babylm_finetune/logs/childes_sft_evals_n0.md``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.dump_scores import (
    dump_regular_scores,
    overlay_regular_section,
)
from ravens_numerical.analysis.experiment_log import TASK_TYPE_ORDER
from ravens_numerical.analysis.plot_corpus_base_vs_sft import (
    is_complete_eval_tasks,
    load_sft_by_category,
    sft_eval_text,
)
from ravens_numerical.models.registry import parse_training_corpus_millions
from ravens_numerical.paths import (
    CHILDES_EXPERIMENTS_MD,
    CHILDES_SFT_ALL_EVALS_MD,
    PLOTS_DIR,
)

_CHANCE_PCT = 25.0
_BUDGET_ORDER = (1, 5, 12, 24)  # millions of words

_BASE_SECTION_RE = re.compile(
    r"^### (.+)\n"
    r"- \*\*Overall:\*\* ([0-9.]+)%[^\n]*\n"
    r"- \*\*By task:\*\* (.+)$",
    re.MULTILINE,
)
_SFT_SECTION_RE = re.compile(
    r"^## `([^`]+)`\s*\n"
    r"- \*\*Overall:\*\* ([0-9.]+)%[^\n]*\n"
    r"- \*\*By task:\*\* (.+)$",
    re.MULTILINE,
)
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")


def _prefer_n_examples_log(path: Path, n_examples: int = 0) -> Path:
    preferred = path.with_name(f"{path.stem}_n{n_examples}{path.suffix}")
    if preferred.is_file():
        return preferred
    return path


def _budget_m(model_or_run_id: str) -> int | None:
    millions = parse_training_corpus_millions(model_or_run_id)
    if millions is None:
        return None
    return int(millions)


def _parse_by_task(line: str) -> dict[str, float]:
    return {task: float(acc) for task, acc in _TASK_ACC_RE.findall(line)}


def load_by_budget(path: Path, section_re: re.Pattern[str]) -> dict[int, dict[str, float]]:
    if not path.is_file():
        raise SystemExit(f"missing log: {path}")
    text = path.read_text(encoding="utf-8")
    out: dict[int, dict[str, float]] = {}
    dump_fill: dict[int, dict[str, float]] = {}
    for match in section_re.finditer(text):
        model_id = match.group(1).strip()
        budget = _budget_m(model_id)
        by_task = _parse_by_task(match.group(3))
        if budget is None or not by_task:
            continue
        if is_complete_eval_tasks(by_task):
            out[budget] = by_task
            continue
        scored = overlay_regular_section(
            text, match.start(), model_id, None, by_task
        )
        if scored is not None:
            _overall, tasks = scored
            if tasks:
                out[budget] = tasks
            continue
        dump = dump_regular_scores(model_id)
        if dump is not None and dump[1]:
            dump_fill.setdefault(budget, dump[1])
    for budget, tasks in dump_fill.items():
        out.setdefault(budget, tasks)
    return out


def plot_subtask_bars(
    base: dict[int, dict[str, float]],
    finetuned: dict[int, dict[str, float]],
    *,
    output: Path,
    title: str,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required") from exc

    budgets = [b for b in _BUDGET_ORDER if b in base or b in finetuned]
    if not budgets:
        raise SystemExit("no CHILDES budgets found in logs")

    tasks = [t for t in TASK_TYPE_ORDER if any(t in (base.get(b) or {}) or t in (finetuned.get(b) or {}) for b in budgets)]
    ncols = 4
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.4 * nrows),
        squeeze=False,
        sharey=True,
    )

    base_color = "#4c78a8"
    sft_color = "#f58518"
    x = np.arange(len(budgets))
    width = 0.36

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        base_vals = [base.get(b, {}).get(task, 0.0) for b in budgets]
        sft_vals = [finetuned.get(b, {}).get(task, 0.0) for b in budgets]
        ax.bar(
            x - width / 2,
            base_vals,
            width,
            color=base_color,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        ax.bar(
            x + width / 2,
            sft_vals,
            width,
            color=sft_color,
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
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}M" for b in budgets], fontsize=9)
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Pretraining words")

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles = [
        Patch(facecolor=base_color, edgecolor="black", label="Base"),
        Patch(facecolor=sft_color, edgecolor="black", label="Finetuned"),
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
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.90, bottom=0.12, hspace=0.4, wspace=0.18
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-log", type=Path, default=CHILDES_EXPERIMENTS_MD)
    parser.add_argument("--sft-log", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="CHILDES GPT-2 ladder: subtask accuracy (base vs Raven SFT)",
    )
    args = parser.parse_args()

    sft_log = args.sft_log or _prefer_n_examples_log(CHILDES_SFT_ALL_EVALS_MD, 0)
    output = args.output or (PLOTS_DIR / "childes_subtask_bars.png")

    base = load_by_budget(args.base_log.resolve(), _BASE_SECTION_RE)
    if args.sft_log is not None:
        finetuned = load_by_budget(sft_log.resolve(), _SFT_SECTION_RE)
    else:
        sft_text = sft_eval_text(CHILDES_SFT_ALL_EVALS_MD, ("childes-",))
        finetuned = load_sft_by_category(sft_text, "words")
    print(f"Base budgets: {sorted(base)}")
    print(f"SFT budgets: {sorted(finetuned)}")

    out = plot_subtask_bars(
        base,
        finetuned,
        output=output.resolve(),
        title=args.title,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
