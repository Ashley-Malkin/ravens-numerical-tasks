#!/usr/bin/env python3
"""Plot scaling curves from experiments.md-style results."""

from __future__ import annotations

import argparse
from pathlib import Path

# 2026-06-23 sweep, n_examples=1 (experiments.md lines 156–234)
PYTHIA = [
    (0.07, 31.7),
    (0.41, 31.7),
    (1.4, 34.2),
    (2.8, 35.8),
    (6.9, 37.5),
]
QWEN3 = [
    (0.6, 25.0),
    (1.7, 35.0),
    (4.0, 60.8),
    (8.0, 60.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "scaling_n_examples_1.png",
    )
    parser.add_argument(
        "--title",
        default="Raven's numerical tasks (n_examples=1, 120 tasks)",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(8, 5))

    px, py = zip(*PYTHIA)
    qx, qy = zip(*QWEN3)

    ax.plot(px, py, "o-", label="Pythia", linewidth=2, markersize=8)
    ax.plot(qx, qy, "s-", label="Qwen3", linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(args.title)
    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
