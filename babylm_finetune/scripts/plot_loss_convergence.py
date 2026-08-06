#!/usr/bin/env python3
"""Plot BabyLM SFT train/eval loss vs epoch from trainer_state.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNS = (
    ("10m", "babylm-10m-gpt2__all_types__20260723T220459Z"),
    ("100m", "babylm-100m-gpt2__all_types__20260723T222637Z"),
)


def load_curves(trainer_state_path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    train: list[tuple[float, float]] = []
    eval_: list[tuple[float, float]] = []
    for entry in data.get("log_history") or []:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "loss" in entry:
            train.append((float(epoch), float(entry["loss"])))
        if "eval_loss" in entry:
            eval_.append((float(epoch), float(entry["eval_loss"])))
    return train, eval_


def plot_convergence(
    runs: list[tuple[str, Path]],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"10m": "#1f77b4", "100m": "#ff7f0e"}

    for label, path in runs:
        train, eval_ = load_curves(path)
        color = colors.get(label, None)
        if train:
            ax.plot(
                [e for e, _ in train],
                [v for _, v in train],
                color=color,
                linestyle="-",
                marker="o",
                markersize=3.5,
                label=f"BabyLM {label} train",
            )
        if eval_:
            ax.plot(
                [e for e, _ in eval_],
                [v for _, v in eval_],
                color=color,
                linestyle="--",
                marker="s",
                markersize=4,
                label=f"BabyLM {label} eval",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("BabyLM SFT loss convergence (all task types)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "outputs",
        help="Directory containing <run_id>/trainer_state.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "plots" / "babylm_sft_loss_convergence.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = []
    for label, run_id in DEFAULT_RUNS:
        path = args.outputs_dir / run_id / "trainer_state.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        runs.append((label, path))

    out = plot_convergence(runs, args.out.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
