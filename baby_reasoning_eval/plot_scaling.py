#!/usr/bin/env python3
"""Plot scaling curves from experiments.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = _EVAL_DIR.parent
BABYLM_EXPERIMENTS_MD = _EVAL_DIR / "babyLMexperiments.md"


def _experiment_log_paths(primary: Path, *, include_babylm_experiments: bool = True) -> list[Path]:
    """Primary log; optionally also ``babyLMexperiments.md``."""
    paths = [primary]
    if (
        include_babylm_experiments
        and primary.resolve() != BABYLM_EXPERIMENTS_MD.resolve()
        and BABYLM_EXPERIMENTS_MD.is_file()
    ):
        paths.append(BABYLM_EXPERIMENTS_MD)
    return paths


def _iter_experiment_blocks(paths: list[Path]):
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        yield from re.split(r"\n---\n", text)


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def parse_experiments_scaling(
    experiments_path: Path,
    *,
    date_prefix: str,
    prompt_type: str = "instruction",
    n_examples: int | None = None,
    prompt_mode: str | None = None,
    include_babylm_experiments: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Return ``(pythia_points, qwen3_points, babylm_points)`` as ``(params_b, accuracy_pct)``."""
    _ensure_repo_on_path()
    from ravens_eval_models import model_family, parse_params_billions

    pythia: list[tuple[float, float]] = []
    qwen3: list[tuple[float, float]] = []
    babylm: list[tuple[float, float]] = []

    for block in _iter_experiment_blocks(
        _experiment_log_paths(
            experiments_path, include_babylm_experiments=include_babylm_experiments
        )
    ):
        header = block.split("\n", 1)[0]
        if date_prefix not in header:
            continue
        if f", {prompt_type})" not in header:
            continue

        if n_examples is not None and f"`--n-examples {n_examples}`" not in block:
            continue
        if prompt_mode is not None and f"`--ravens-prompt-mode {prompt_mode}`" not in block:
            continue

        model_match = re.search(r"^### (.+)$", block, re.MULTILINE)
        overall_match = re.search(
            r"\*\*Overall:\*\* ([0-9.]+)%", block
        )
        if not model_match or not overall_match:
            continue

        model_id = model_match.group(1).strip()
        params_b = parse_params_billions(model_id)
        if params_b is None:
            continue
        accuracy = float(overall_match.group(1))
        point = (params_b, accuracy)
        family = model_family(model_id)
        if family == "pythia":
            pythia.append(point)
        elif family == "qwen3":
            qwen3.append(point)
        elif family == "babylm":
            babylm.append(point)

    pythia.sort(key=lambda x: x[0])
    qwen3.sort(key=lambda x: x[0])
    babylm.sort(key=lambda x: x[0])
    return pythia, qwen3, babylm


def plot_scaling(
    pythia: list[tuple[float, float]],
    qwen3: list[tuple[float, float]],
    *,
    babylm: list[tuple[float, float]] | None = None,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))

    if pythia:
        px, py = zip(*pythia)
        ax.plot(
            px, py, "o-", label="Pythia", color="tab:blue", linewidth=2, markersize=8
        )
    if qwen3:
        qx, qy = zip(*qwen3)
        ax.plot(
            qx, qy, "s-", label="Qwen3", color="tab:orange", linewidth=2, markersize=8
        )
    babylm = babylm or []
    if babylm:
        bx, by = zip(*babylm)
        ax.plot(
            bx, by, "^-", label="BabyLM", color="tab:green", linewidth=2, markersize=8
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling from experiments.md")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=_EVAL_DIR / "experiments.md",
    )
    parser.add_argument(
        "--date",
        default="2026-06-29",
        help="Filter entries whose header date starts with this (default: 2026-06-29)",
    )
    parser.add_argument(
        "--prompt-type",
        default="instruction",
        choices=("instruction", "completion"),
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=1,
        help="Filter by --n-examples in log line (default: 1)",
    )
    parser.add_argument(
        "--prompt-mode",
        default="choice_only",
        help="For instruction runs, filter by ravens-prompt-mode (default: choice_only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--title",
        default=None,
    )
    parser.add_argument(
        "--family",
        choices=("pythia", "qwen3", "babylm"),
        default=None,
        help="Plot only one model family (default: all present)",
    )
    parser.add_argument(
        "--no-babylm-experiments",
        action="store_true",
        help="Do not merge babyLMexperiments.md (Pythia/Qwen3 from experiments.md only)",
    )
    args = parser.parse_args()

    prompt_mode = args.prompt_mode if args.prompt_type == "instruction" else None
    pythia, qwen3, babylm = parse_experiments_scaling(
        args.experiments,
        date_prefix=args.date,
        prompt_type=args.prompt_type,
        n_examples=args.n_examples,
        prompt_mode=prompt_mode,
        include_babylm_experiments=not args.no_babylm_experiments,
    )
    if args.family == "pythia":
        qwen3 = []
        babylm = []
    elif args.family == "qwen3":
        pythia = []
        babylm = []
    elif args.family == "babylm":
        pythia = []
        qwen3 = []
    if args.no_babylm_experiments:
        babylm = []
    if not pythia and not qwen3 and not babylm:
        raise SystemExit(
            f"No matching entries in {args.experiments} "
            f"(date={args.date}, prompt_type={args.prompt_type})"
        )

    suffix = f"{args.prompt_type}_{args.date}"
    output = args.output or (_EVAL_DIR / f"scaling_{suffix}.png")
    title = args.title or (
        f"Raven's numerical tasks — {args.prompt_type} "
        f"(n_examples={args.n_examples}, {args.date})"
    )
    plot_scaling(pythia, qwen3, babylm=babylm, output=output, title=title)


if __name__ == "__main__":
    main()
