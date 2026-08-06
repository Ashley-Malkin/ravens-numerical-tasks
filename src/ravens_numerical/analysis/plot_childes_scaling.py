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

from ravens_numerical.models.registry import parse_training_corpus_millions
from ravens_numerical.paths import (
    CHILDES_EXPERIMENTS_MD,
    CHILDES_SFT_ALL_EVALS_MD,
    PLOTS_DIR,
)

_CHANCE_PCT = 25.0

_BASE_SECTION_RE = re.compile(
    r"^### (.+)\n"
    r"- \*\*Overall:\*\* ([0-9.]+)%",
    re.MULTILINE,
)
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
    points: list[tuple[int, float, str]] = []
    for match in _BASE_SECTION_RE.finditer(text):
        model_id = match.group(1).strip()
        words = _words_from_id(model_id)
        if words is None:
            continue
        points.append((words, float(match.group(2)), model_id))
    return sorted(points, key=lambda item: item[0])


def load_sft_points(path: Path) -> list[tuple[int, float, str]]:
    if not path.is_file():
        raise SystemExit(f"missing finetuned log: {path}")
    text = path.read_text(encoding="utf-8")
    points: list[tuple[int, float, str]] = []
    for match in _SFT_SECTION_RE.finditer(text):
        run_id = match.group(1).strip()
        words = _words_from_id(run_id)
        if words is None:
            continue
        points.append((words, float(match.group(2)), run_id))
    return sorted(points, key=lambda item: item[0])


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
    finetuned = load_sft_points(sft_log.resolve())
    if not base and not finetuned:
        raise SystemExit("no CHILDES points found in either log")

    print(f"Base points ({len(base)}): {[(w, a) for w, a, _ in base]}")
    print(f"SFT points ({len(finetuned)}): {[(w, a) for w, a, _ in finetuned]}")
    out = plot_childes(base, finetuned, output=output.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
