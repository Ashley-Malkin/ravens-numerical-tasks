#!/usr/bin/env python3
"""Recompute Raven forced-choice accuracy from stored trial JSON.

Applies the grading rule used by ``evaluate`` for ``forced_choice``:

1. If echo ``answer_logprobs`` have a unique length-normalized argmax
   (spread at least ``min_spread``) → that choice decides correctness.
2. Else if generation matches a choice → use generation (echo abstained).
3. Else count incorrect. Collapsed near-zero / tied echo scores abstain.

Example::

    PYTHONPATH=src python3 babylm_finetune/scripts/recompute_forced_choice_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ravens_numerical.scoring.echo_logprobs import unique_logprob_argmax  # noqa: E402


def _match_generation(row: dict) -> str | None:
    text = (row.get("response") or {}).get("text") or ""
    span = text.split("]")[0].strip().lower()
    choices = (row.get("stimulus") or {}).get("answer_choices") or []
    if not choices:
        return None
    choice_map = {c.lower(): c for c in choices}
    formatted = {(c + "]").lower(): c for c in choices}
    if span in formatted:
        return formatted[span]
    bare = span.rstrip("]").strip()
    if bare in choice_map:
        return choice_map[bare]
    if span in choice_map:
        return choice_map[span]
    return None


def analyze_trials(data: list[dict]) -> dict:
    n = len(data)
    logged = gen = fixed = gen_valid = echo_scored = echo_correct = all_zero = 0
    for row in data:
        exp = row["stimulus"]["expected"]
        if row.get("score", {}).get("correct"):
            logged += 1
        matched = _match_generation(row)
        if matched is not None:
            gen_valid += 1
            if matched == exp:
                gen += 1
        lp = row.get("score", {}).get("answer_logprobs") or {}
        valid = {c: v for c, v in lp.items() if v is not None}
        if valid and all(v == 0 for v in valid.values()):
            all_zero += 1
        pred = unique_logprob_argmax(valid) if valid else None
        if pred is not None:
            echo_scored += 1
            if pred == exp:
                echo_correct += 1
        if pred is not None:
            ok = pred == exp
        elif matched is not None:
            ok = matched == exp
        else:
            ok = False
        if ok:
            fixed += 1
    return {
        "n": n,
        "logged_acc": logged / n if n else 0.0,
        "fixed_fc_acc": fixed / n if n else 0.0,
        "gen_match_acc": gen / n if n else 0.0,
        "gen_in_choices": gen_valid / n if n else 0.0,
        "echo_unique_acc": (echo_correct / echo_scored) if echo_scored else None,
        "echo_unique_n": echo_scored,
        "all_zero_frac": all_zero / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs",
        help="Root of exported Modal / local run trees",
    )
    parser.add_argument(
        "--filter",
        default="checkpoints|babylm|BabyLM",
        help="Regex matched against model directory names (default: SFT + BabyLM)",
    )
    args = parser.parse_args()
    import re

    pat = re.compile(args.filter)
    runs_dir: Path = args.runs_dir
    print(
        f"{'model':<68} {'n':>2} {'logged':>7} {'fixed':>7} "
        f"{'gen':>7} {'inCh%':>6} {'zero%':>6}"
    )
    print("-" * 110)
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir() or not pat.search(model_dir.name):
            continue
        by_n: dict[int, Path] = {}
        for p in model_dir.glob("*/ravens_numerical/*examples*.json"):
            try:
                n_ex = int(p.name.split("_", 1)[0])
            except ValueError:
                continue
            if n_ex not in by_n or p.stat().st_mtime > by_n[n_ex].stat().st_mtime:
                by_n[n_ex] = p
        for n_ex, path in sorted(by_n.items()):
            stats = analyze_trials(json.loads(path.read_text(encoding="utf-8")))
            short = model_dir.name.replace("--checkpoints--", "")[:66]
            print(
                f"{short:<68} {n_ex:>2} {100 * stats['logged_acc']:6.1f}% "
                f"{100 * stats['fixed_fc_acc']:6.1f}% "
                f"{100 * stats['gen_match_acc']:6.1f}% "
                f"{100 * stats['gen_in_choices']:5.1f}% "
                f"{100 * stats['all_zero_frac']:5.1f}%"
            )


if __name__ == "__main__":
    main()
