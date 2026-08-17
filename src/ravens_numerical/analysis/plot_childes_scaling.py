#!/usr/bin/env python3
"""Plot CHILDES ladder overall accuracy vs pretraining word budget.

Sources:

- Base: ``artifacts/logs/childesExperiments.md``
- Finetuned: ``babylm_finetune/logs/childes_sft_evals_n0.md``
  (falls back to unsuffixed ``childes_sft_evals.md``)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ravens_numerical.analysis.dump_scores import (
    dump_regular_scores,
    overlay_regular_section,
)
from ravens_numerical.analysis.plot_corpus_base_vs_sft import (
    _BASE_SECTION_RE,
    easy_overall,
    is_complete_eval_tasks,
    is_complete_sft_run,
    iter_sft_sections,
    sft_eval_text,
)
from ravens_numerical.models.registry import parse_training_corpus_millions
from ravens_numerical.paths import (
    CHILDES_EXPERIMENTS_MD,
    CHILDES_SFT_ALL_EVALS_MD,
    PLOTS_DIR,
)

_CHANCE_PCT = 25.0
_TASK_ACC_RE = re.compile(r"([a-z_]+) ([0-9.]+)%")
_SFT_SECTION_RE = re.compile(
    r"^## `([^`]+)`\s*\n"
    r"- \*\*Overall:\*\* ([0-9.]+)%",
    re.MULTILINE,
)


def _prefer_n_examples_log(path: Path, n_examples: int = 0) -> Path:
    preferred = path.with_name(f"{path.stem}_n{n_examples}{path.suffix}")
    if preferred.is_file():
        return preferred
    return path


def _words_from_id(model_or_run_id: str) -> int | None:
    millions = parse_training_corpus_millions(model_or_run_id)
    if millions is None:
        return None
    return int(millions * 1_000_000)


def load_base_points(path: Path) -> list[tuple[int, float, str]]:
    if not path.is_file():
        raise SystemExit(f"missing base log: {path}")
    text = path.read_text(encoding="utf-8")
    points: dict[int, tuple[float, str]] = {}
    dump_fill: dict[int, tuple[float, str]] = {}
    for match in _BASE_SECTION_RE.finditer(text):
        model_id = match.group(1).strip()
        words = _words_from_id(model_id)
        if words is None:
            continue
        by_task = {
            task: float(acc) for task, acc in _TASK_ACC_RE.findall(match.group(3))
        } if match.lastindex and match.lastindex >= 3 else {}
        if is_complete_eval_tasks(by_task):
            points[words] = (
                easy_overall(float(match.group(2)), by_task),
                model_id,
            )
            continue
        scored = overlay_regular_section(
            text, match.start(), model_id, float(match.group(2)), by_task
        )
        if scored is not None:
            overall, _ = scored
            points[words] = (float(overall), model_id)
            continue
        dump = dump_regular_scores(model_id)
        if dump is not None:
            dump_fill.setdefault(words, (float(dump[0]), model_id))
    for words, value in dump_fill.items():
        points.setdefault(words, value)
    return sorted(
        [(w, acc, mid) for w, (acc, mid) in points.items()],
        key=lambda item: item[0],
    )


def load_sft_points(path: Path, text: str | None = None) -> list[tuple[int, float, str]]:
    if text is None:
        if not path.is_file():
            raise SystemExit(f"missing finetuned log: {path}")
        text = path.read_text(encoding="utf-8")
    complete: dict[int, tuple[float, str]] = {}
    other: dict[int, tuple[float, str]] = {}
    for run_id, logged, by_task, _start in iter_sft_sections(text):
        words = _words_from_id(run_id)
        if words is None:
            continue
        acc = easy_overall(logged, by_task)
        bucket = complete if is_complete_sft_run(run_id) else other
        bucket[words] = (acc, run_id)
    points = complete or other
    if points:
        return sorted(
            [(w, acc, mid) for w, (acc, mid) in points.items()],
            key=lambda item: item[0],
        )
    # Fallback: combined-log ## `run_id` overall-only sections (no by-task).
    if not path.is_file() and text is None:
        raise SystemExit(f"missing finetuned log: {path}")
    fallback_text = text if text is not None else path.read_text(encoding="utf-8")
    dump_fill: dict[int, tuple[float, str]] = {}
    points2: dict[int, tuple[float, str]] = {}
    for match in _SFT_SECTION_RE.finditer(fallback_text):
        run_id = match.group(1).strip()
        words = _words_from_id(run_id)
        if words is None:
            continue
        scored = overlay_regular_section(
            fallback_text, match.start(), run_id, float(match.group(2)), {}
        )
        if scored is not None:
            overall, _ = scored
            points2[words] = (float(overall), run_id)
            continue
        dump = dump_regular_scores(run_id)
        if dump is not None:
            dump_fill.setdefault(words, (float(dump[0]), run_id))
    for words, value in dump_fill.items():
        points2.setdefault(words, value)
    return sorted(
        [(w, acc, mid) for w, (acc, mid) in points2.items()],
        key=lambda item: item[0],
    )


def plot_childes(
    base: list[tuple[int, float, str]],
    finetuned: list[tuple[int, float, str]],
    *,
    output: Path,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib required: pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    if base:
        ax.plot(
            [w for w, _, _ in base],
            [acc for _, acc, _ in base],
            "o-",
            color="#4c78a8",
            linewidth=2.2,
            markersize=8,
            label="Base",
        )
    if finetuned:
        ax.plot(
            [w for w, _, _ in finetuned],
            [acc for _, acc, _ in finetuned],
            "s-",
            color="#f58518",
            linewidth=2.2,
            markersize=8,
            label="Finetuned",
        )

    ax.axhline(
        _CHANCE_PCT,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Chance (25%)",
    )
    ax.set_xlabel("Pretraining words")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("CHILDES GPT-2 ladder (seed42): base vs Raven SFT")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    tick_words = sorted({w for w, _, _ in base + finetuned})
    if tick_words:
        ax.set_xticks(tick_words)
        ax.set_xticklabels(
            [f"{w // 1_000_000}M" for w in tick_words],
            fontsize=9,
        )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-log",
        type=Path,
        default=CHILDES_EXPERIMENTS_MD,
        help="Base CHILDES eval markdown",
    )
    parser.add_argument(
        "--sft-log",
        type=Path,
        default=None,
        help="Finetuned CHILDES SFT eval markdown (default: childes_sft_evals_n0.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    args = parser.parse_args()

    sft_log = args.sft_log or _prefer_n_examples_log(CHILDES_SFT_ALL_EVALS_MD, 0)
    output = args.output or (PLOTS_DIR / "childes_base_vs_sft_scaling.png")

    base = load_base_points(args.base_log.resolve())
    sft_text = (
        args.sft_log.resolve().read_text(encoding="utf-8")
        if args.sft_log is not None
        else sft_eval_text(CHILDES_SFT_ALL_EVALS_MD, ("childes-",))
    )
    finetuned = load_sft_points(sft_log.resolve(), text=sft_text)
    if not base and not finetuned:
        raise SystemExit("no CHILDES points found in either log")

    print(f"Base points ({len(base)}): {[(w, a) for w, a, _ in base]}")
    print(f"SFT points ({len(finetuned)}): {[(w, a) for w, a, _ in finetuned]}")
    out = plot_childes(base, finetuned, output=output.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
