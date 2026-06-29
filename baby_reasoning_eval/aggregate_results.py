#!/usr/bin/env python3
"""Aggregate ravens_numerical JSON results into CSV and a scaling summary table.

Walks ``baby-reasoning/results/`` for ``ravens_numerical/*_examples.json`` files.

Usage (from repo root):

    python baby_reasoning_eval/aggregate_results.py
    python baby_reasoning_eval/aggregate_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = _EVAL_DIR.parent
CONTAINER_RAVENS_ROOT = Path("/root/ravens")


def _ensure_ravens_repo_on_path() -> None:
    for root in (REPO_ROOT, CONTAINER_RAVENS_ROOT):
        if (root / "ravens_eval_models.py").is_file():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return
    raise ImportError(
        f"Could not find ravens_eval_models.py under {REPO_ROOT} or {CONTAINER_RAVENS_ROOT}"
    )


_ensure_ravens_repo_on_path()
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from ravens_eval_models import (  # noqa: E402
    model_family,
    parse_params_billions,
)
from experiment_log import TASK_TYPE_ORDER  # noqa: E402

DEFAULT_RESULTS_DIR = _EVAL_DIR / "baby-reasoning" / "results"
DEFAULT_CSV = _EVAL_DIR / "scaling_results.csv"
DEFAULT_MD = _EVAL_DIR / "scaling_results.md"

_EXAMPLES_RE = re.compile(
    r"^(\d+)_examples(?:_(instruction|completion|choice_only))?\.json$"
)


def _parse_n_examples(path: Path) -> int | None:
    m = _EXAMPLES_RE.match(path.name)
    return int(m.group(1)) if m else None


def _parse_prompt_type(path: Path) -> str:
    m = _EXAMPLES_RE.match(path.name)
    if not m:
        return "instruction"
    return m.group(2) or "instruction"


def _model_id_from_tag(model_tag: str) -> str:
    """Reverse ``save_results`` model tag to HF id (best effort)."""
    return model_tag.replace("--", "/")


def _summarize_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")

    by_type: dict[str, list[bool]] = defaultdict(list)
    letter_counts: Counter[str] = Counter()
    parse_failures = 0

    for row in data:
        correct = bool(row.get("score", {}).get("correct", False))
        task_type = (
            row.get("stimulus", {})
            .get("metadata", {})
            .get("task", {})
            .get("task_type", "unknown")
        )
        by_type[str(task_type)].append(correct)

        text = (row.get("response") or {}).get("text", "")
        letters = re.findall(r"\b([A-D])\b", text.upper())
        if letters:
            letter_counts[letters[-1]] += 1
        elif not correct:
            parse_failures += 1

    total = len(data)
    n_correct = sum(1 for row in data if row.get("score", {}).get("correct"))
    row: dict = {
        "results_path": str(path),
        "total": total,
        "correct": n_correct,
        "accuracy": n_correct / total if total else 0.0,
        "parse_failures": parse_failures,
        "majority_letter": letter_counts.most_common(1)[0][0] if letter_counts else "",
        "majority_letter_pct": (
            letter_counts.most_common(1)[0][1] / total if letter_counts and total else 0.0
        ),
    }
    brier_vals = [
        r.get("score", {}).get("brier")
        for r in data
        if r.get("score", {}).get("brier") is not None
    ]
    if brier_vals:
        row["mean_brier"] = sum(brier_vals) / len(brier_vals)
        row["brier_n"] = len(brier_vals)
    for task_type in TASK_TYPE_ORDER:
        vals = by_type.get(task_type, [])
        row[f"acc_{task_type}"] = sum(vals) / len(vals) if vals else None
    for task_type, vals in sorted(by_type.items()):
        if task_type not in TASK_TYPE_ORDER:
            row[f"acc_{task_type}"] = sum(vals) / len(vals) if vals else None
    return row


def collect_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("**/ravens_numerical/*.json")):
        n_examples = _parse_n_examples(path)
        if n_examples is None:
            continue

        # results/<model_tag>/<run_id>/ravens_numerical/N_examples.json
        parts = path.relative_to(results_dir).parts
        if len(parts) < 4:
            continue
        model_tag = parts[0]
        run_id = parts[1]
        model_id = _model_id_from_tag(model_tag)

        summary = _summarize_json(path)
        rows.append(
            {
                "model": model_id,
                "model_tag": model_tag,
                "run_id": run_id,
                "family": model_family(model_id),
                "params_b": parse_params_billions(model_id),
                "n_examples": n_examples,
                "prompt_type": _parse_prompt_type(path),
                **summary,
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    base_fields = [
        "model",
        "family",
        "params_b",
        "n_examples",
        "prompt_type",
        "accuracy",
        "correct",
        "total",
        "parse_failures",
        "majority_letter",
        "majority_letter_pct",
        "run_id",
        "results_path",
    ]
    task_fields = sorted(
        k for k in rows[0] if k.startswith("acc_") and k not in base_fields
    )
    fieldnames = base_fields + task_fields

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["family"], r["params_b"] or 0, r["n_examples"])):
            writer.writerow(row)


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def write_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Ravens numerical scaling results",
        "",
        "Generated by `baby_reasoning_eval/aggregate_results.py`.",
        "",
        "Chance baseline for 4-way MCQ: **25.0%**.",
        "",
    ]

    by_setting: dict[tuple[int], list[dict]] = defaultdict(list)
    for row in rows:
        by_setting[(row["n_examples"],)].append(row)

    for (n_examples,) in sorted(by_setting):
        lines.append(f"## n_examples={n_examples}")
        lines.append("")
        for family in ("pythia", "qwen3"):
            family_rows = [
                r for r in by_setting[(n_examples,)] if r["family"] == family
            ]
            if not family_rows:
                continue
            lines.append(f"### {family}")
            lines.append("")
            lines.append("| Model | Params (B) | Overall | Correct |")
            lines.append("|-------|------------|---------|---------|")
            for r in sorted(family_rows, key=lambda x: x["params_b"] or 0):
                params = f"{r['params_b']:.3g}" if r["params_b"] is not None else "?"
                lines.append(
                    f"| {r['model']} | {params} | {_pct(r['accuracy'])} | "
                    f"{r['correct']}/{r['total']} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ravens_numerical eval JSON results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Root results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MD,
        help=f"Output markdown path (default: {DEFAULT_MD})",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"No results directory: {args.results_dir} (writing empty outputs)", file=sys.stderr)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv([], args.csv)
        write_markdown([], args.markdown)
        return

    rows = collect_results(args.results_dir)
    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print(f"Wrote {len(rows)} rows → {args.csv}")
    print(f"Wrote scaling table → {args.markdown}")


if __name__ == "__main__":
    main()
