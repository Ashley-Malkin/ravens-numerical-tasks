#!/usr/bin/env python3
"""Plot BabyLM SFT N-scaling: test accuracy vs training set size N."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ravens_numerical.analysis.dump_scores import (  # noqa: E402
    dump_regular_scores,
    overlay_regular_section,
)
from ravens_numerical.paths import (  # noqa: E402
    BABYLM_SFT_N_SCALING_EVALS_MD,
    BABYLM_SFT_N_SCALING_MANIFEST,
    BABYLM_SFT_N_SCALING_RESULTS,
)

_SECTION_RE = re.compile(
    r"^## `([^`]+)`\s*\n"
    r"- \*\*Overall:\*\* ([0-9.]+)%",
    re.MULTILINE,
)


def _scale_label(model_tag: str) -> str:
    if "100m" in model_tag:
        return "100m"
    if "10m" in model_tag:
        return "10m"
    return model_tag


def load_results_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"unexpected results JSON shape in {path}")
    out: list[dict] = []
    for row in rows:
        if row.get("N") is None or row.get("accuracy") is None:
            continue
        run_id = row.get("run_id")
        if run_id:
            dump = dump_regular_scores(str(run_id))
            if dump is not None:
                row = {**row, "accuracy": float(dump[0]) / 100.0}
        out.append(row)
    return out


def load_from_md_and_manifest(
    md_path: Path,
    manifest_path: Path,
) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    acc_by_id = {}
    dump_fill = {}
    for match in _SECTION_RE.finditer(text):
        run_id = match.group(1)
        scored = overlay_regular_section(
            text, match.start(), run_id, float(match.group(2)), {}
        )
        if scored is not None:
            acc_by_id[run_id] = float(scored[0]) / 100.0
            continue
        dump = dump_regular_scores(run_id)
        if dump is not None:
            dump_fill.setdefault(run_id, float(dump[0]) / 100.0)
    for run_id, acc in dump_fill.items():
        acc_by_id.setdefault(run_id, acc)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for entry in manifest.get("runs") or []:
        run_id = entry.get("run_id")
        if not run_id or run_id not in acc_by_id:
            continue
        rows.append(
            {
                "run_id": run_id,
                "model_tag": entry.get("model_tag") or run_id.split("__", 1)[0],
                "N": int(entry["N"]),
                "accuracy": acc_by_id[run_id],
            }
        )
    return rows


def plot_n_scaling(rows: list[dict], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    by_scale: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for row in rows:
        scale = _scale_label(str(row["model_tag"]))
        by_scale[scale].append((int(row["N"]), float(row["accuracy"]), str(row["run_id"])))

    if not by_scale:
        raise SystemExit("no N-scaling rows to plot")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scale in ("10m", "100m"):
        pts = sorted(by_scale.get(scale, []), key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [100.0 * p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"BabyLM {scale}")

    ax.set_xlabel("SFT train size N")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("BabyLM SFT N-scaling (all task types)")
    ax.set_xticks(sorted({int(r["N"]) for r in rows}))
    ax.set_ylim(0, 100)
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
        "--results-json",
        type=Path,
        default=BABYLM_SFT_N_SCALING_RESULTS,
        help="Preferred input: n_scaling_results.json",
    )
    parser.add_argument(
        "--evals-md",
        type=Path,
        default=BABYLM_SFT_N_SCALING_EVALS_MD,
        help="Fallback markdown if JSON missing",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BABYLM_SFT_N_SCALING_MANIFEST,
        help="Manifest used with --evals-md fallback",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "babylm_finetune" / "outputs" / "n_scaling_accuracy.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    if args.results_json.is_file():
        rows = load_results_json(args.results_json)
        print(f"Loaded {len(rows)} rows from {args.results_json}")
    elif args.evals_md.is_file() and args.manifest.is_file():
        rows = load_from_md_and_manifest(args.evals_md, args.manifest)
        print(f"Loaded {len(rows)} rows from {args.evals_md} + {args.manifest}")
    else:
        raise SystemExit(
            "Need n_scaling_results.json or (n_scaling_evals.md + n_scaling_manifest.json)"
        )

    out = plot_n_scaling(rows, args.out.resolve())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
