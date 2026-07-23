#!/usr/bin/env python3
"""Audit instruction choice_only results for parsing vs model errors.

Compares text parsing (``parse_answer``) against logprob argmax (independent of
parser), checks value-aligned letter mapping, validates ``tasks.json`` labels,
and optionally cross-checks instruction vs completion runs on the same tasks.

Usage (from repo root):

    python baby_reasoning_eval/audit_instruction_results.py \\
        path/to/pythia_1_examples_choice_only.json \\
        path/to/qwen3_base_1_examples_choice_only.json

    python baby_reasoning_eval/audit_instruction_results.py \\
        --validate-tasks tasks.json

    python baby_reasoning_eval/audit_instruction_results.py \\
        --instruction path/to/instruction.json \\
        --completion path/to/completion.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ravens_numerical.parsing.answer_parse import parse_answer, parse_structured_choice
from ravens_numerical.paths import TASKS_JSON
from ravens_numerical.prompts.prompts import (
    expected_completion_answer,
    format_completion_answer,
)
from ravens_numerical.scoring.choice_logprobs import parse_logprobs_by_letter_vllm


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _task_key(task: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_type": task.get("task_type"),
            "matrix": task.get("matrix"),
            "answer_options": task.get("answer_options"),
        },
        sort_keys=True,
    )


def _letter_to_value(task: dict[str, Any], letter_idx: int | None) -> str | None:
    if letter_idx is None or not (0 <= letter_idx < 4):
        return None
    return format_completion_answer(task["answer_options"][letter_idx])


def _value_match(predicted: str | None, expected: str, task_type: str) -> bool:
    if predicted is None:
        return False
    if task_type == "intersection":
        return set(predicted.split()) == set(expected.split())
    return predicted == expected


@dataclass
class TrialAudit:
    task_type: str
    correct_index: int
    text: str
    pred_text: int | None
    pred_logprob: int | None
    stored_correct: bool | None
    stored_logprob_correct: bool | None
    parse_path: str
    value_aligned: bool
    text_correct: bool
    logprob_correct: bool | None
    disagree: bool


@dataclass
class AuditReport:
    path: Path
    total: int = 0
    text_correct: int = 0
    logprob_correct: int = 0
    logprob_available: int = 0
    value_aligned: int = 0
    parse_none: int = 0
    structured_json: int = 0
    fallback_parse: int = 0
    stored_mismatch: int = 0
    text_logprob_disagree: int = 0
    by_task_type: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    disagreements: list[TrialAudit] = field(default_factory=list)
    parse_failures: list[TrialAudit] = field(default_factory=list)


def audit_trial(row: dict[str, Any]) -> TrialAudit:
    task = row["stimulus"]["metadata"]["task"]
    ci = int(task["correct_index"])
    text = (row.get("response") or {}).get("text", "")
    raw = (row.get("response") or {}).get("raw")
    score = row.get("score") or {}

    pred_text = parse_answer(text, task["answer_options"], ci)
    pred_structured = parse_structured_choice(text)
    parse_path = "structured_json" if pred_structured is not None else (
        "fallback" if pred_text is not None else "none"
    )

    pred_logprob: int | None = None
    if raw:
        pred_logprob, _ = parse_logprobs_by_letter_vllm(raw, text)

    expected_value = expected_completion_answer(task)
    pred_value = _letter_to_value(task, pred_text)
    value_aligned = _value_match(pred_value, expected_value, task["task_type"])

    text_correct = pred_text == ci
    logprob_correct = pred_logprob == ci if pred_logprob is not None else None
    stored_correct = score.get("correct")
    stored_logprob = score.get("logprob_argmax_correct")

    disagree = (
        logprob_correct is not None
        and text_correct != logprob_correct
    )

    return TrialAudit(
        task_type=str(task.get("task_type", "unknown")),
        correct_index=ci,
        text=text,
        pred_text=pred_text,
        pred_logprob=pred_logprob,
        stored_correct=stored_correct,
        stored_logprob_correct=stored_logprob,
        parse_path=parse_path,
        value_aligned=value_aligned,
        text_correct=text_correct,
        logprob_correct=logprob_correct,
        disagree=disagree,
    )


def audit_results_file(path: Path, *, max_samples: int = 15) -> AuditReport:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")

    report = AuditReport(path=path, total=len(rows))

    for row in rows:
        trial = audit_trial(row)
        tt = trial.task_type
        bucket = report.by_task_type[tt]
        bucket["total"] = bucket.get("total", 0) + 1

        if trial.text_correct:
            report.text_correct += 1
            bucket["text_correct"] = bucket.get("text_correct", 0) + 1
        if trial.value_aligned:
            report.value_aligned += 1
            bucket["value_aligned"] = bucket.get("value_aligned", 0) + 1
        if trial.pred_logprob is not None:
            report.logprob_available += 1
            if trial.logprob_correct:
                report.logprob_correct += 1
                bucket["logprob_correct"] = bucket.get("logprob_correct", 0) + 1
        if trial.pred_text is None:
            report.parse_none += 1
            if len(report.parse_failures) < max_samples:
                report.parse_failures.append(trial)
        if trial.parse_path == "structured_json":
            report.structured_json += 1
        elif trial.parse_path == "fallback":
            report.fallback_parse += 1
        if trial.stored_correct is not None and trial.stored_correct != trial.text_correct:
            report.stored_mismatch += 1
        if trial.disagree:
            report.text_logprob_disagree += 1
            if len(report.disagreements) < max_samples:
                report.disagreements.append(trial)

    return report


def validate_tasks_json(path: Path, *, max_tasks: int | None = None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("tasks", data)
    issues: list[str] = []
    letters = "ABCD"

    for i, task in enumerate(raw):
        if max_tasks is not None and i >= max_tasks:
            break
        if not isinstance(task, dict):
            issues.append(f"task[{i}]: not a dict")
            continue
        ci = int(task["correct_index"])
        letter = task.get("correct_letter") or letters[ci]
        expected_letter = letters[ci]
        if str(letter).strip().upper()[:1] != expected_letter:
            issues.append(
                f"task[{i}] ({task.get('task_type')}): "
                f"correct_letter={letter!r} vs ABCD[{ci}]={expected_letter!r}"
            )
        opts = task.get("answer_options", [])
        if not (0 <= ci < len(opts)):
            issues.append(f"task[{i}]: correct_index={ci} out of range for options")
    return issues


def run_oracle_check(max_tasks: int = 140) -> bool:
    from ravens_numerical.eval.tasks.base import ModelResponse
    from ravens_numerical.eval.tasks.ravens import RavensNumericalTask

    task = RavensNumericalTask(
        max_tasks=max_tasks,
        prompt_mode="choice_only",
    )
    ok = True
    for stimulus in task.canonical_stimuli():
        letter = stimulus.expected
        response = ModelResponse(text=f'{{"choice":"{letter}"}}')
        if not task.score(response, stimulus):
            ok = False
            print(f"Oracle FAIL: letter {letter} for {stimulus.metadata['task']['task_type']}")
    return ok


@dataclass
class CrossFormatReport:
    instruction_path: Path
    completion_path: Path
    matched: int = 0
    instruction_only: int = 0
    completion_only: int = 0
    both_wrong: int = 0
    letter_wrong_value_right: int = 0


def compare_instruction_completion(
    instruction_path: Path,
    completion_path: Path,
) -> CrossFormatReport:
    inst_rows = json.loads(instruction_path.read_text(encoding="utf-8"))
    comp_rows = json.loads(completion_path.read_text(encoding="utf-8"))

    inst_by_key = {
        _task_key(r["stimulus"]["metadata"]["task"]): r for r in inst_rows
    }
    comp_by_key = {
        _task_key(r["stimulus"]["metadata"]["task"]): r for r in comp_rows
    }

    report = CrossFormatReport(
        instruction_path=instruction_path,
        completion_path=completion_path,
    )

    for key, inst_row in inst_by_key.items():
        comp_row = comp_by_key.get(key)
        if comp_row is None:
            continue

        inst_trial = audit_trial(inst_row)
        comp_correct = bool(comp_row.get("score", {}).get("correct", False))
        inst_correct = inst_trial.text_correct

        if inst_correct and comp_correct:
            report.matched += 1
        elif inst_correct:
            report.instruction_only += 1
        elif comp_correct:
            report.completion_only += 1
        else:
            report.both_wrong += 1

        if comp_correct and not inst_correct and inst_trial.value_aligned:
            report.letter_wrong_value_right += 1

    return report


def _format_report(report: AuditReport) -> str:
    lines = [
        f"## {report.path.name}",
        f"- path: `{report.path}`",
        f"- trials: {report.total}",
        "",
        "### Accuracy (recomputed)",
        f"- text parse (`parse_answer`): {_pct(report.text_correct, report.total)} "
        f"({report.text_correct}/{report.total})",
        f"- logprob argmax: {_pct(report.logprob_correct, report.logprob_available)} "
        f"({report.logprob_correct}/{report.logprob_available} with logprobs)",
        f"- value-aligned (letter → option value): "
        f"{_pct(report.value_aligned, report.total)} "
        f"({report.value_aligned}/{report.total})",
        "",
        "### Parsing",
        f"- structured JSON path: {report.structured_json}/{report.total}",
        f"- fallback regex path: {report.fallback_parse}/{report.total}",
        f"- parse failures (pred=None): {report.parse_none}/{report.total}",
        f"- stored `score.correct` ≠ recomputed: {report.stored_mismatch}/{report.total}",
        f"- text vs logprob disagree: {report.text_logprob_disagree}/{report.total}",
        "",
        "### By task type (text parse accuracy)",
    ]
    for tt in sorted(report.by_task_type):
        b = report.by_task_type[tt]
        total = b.get("total", 0)
        tc = b.get("text_correct", 0)
        va = b.get("value_aligned", 0)
        lines.append(
            f"- {tt}: text {_pct(tc, total)} · value-aligned {_pct(va, total)} (n={total})"
        )

    if report.disagreements:
        lines.extend(["", "### Sample text/logprob disagreements"])
        for t in report.disagreements[:5]:
            lines.append(
                f"- [{t.task_type}] text={t.pred_text} logprob={t.pred_logprob} "
                f"correct={t.correct_index} response={t.text!r}"
            )

    if report.parse_failures:
        lines.extend(["", "### Sample parse failures"])
        for t in report.parse_failures[:5]:
            lines.append(f"- [{t.task_type}] response={t.text!r}")

    lines.append("")
    return "\n".join(lines)


def _format_cross(report: CrossFormatReport) -> str:
    total = (
        report.matched
        + report.instruction_only
        + report.completion_only
        + report.both_wrong
    )
    return "\n".join(
        [
            "## Instruction vs completion (same tasks)",
            f"- instruction: `{report.instruction_path.name}`",
            f"- completion: `{report.completion_path.name}`",
            f"- paired tasks: {total}",
            f"- both correct: {report.matched} ({_pct(report.matched, total)})",
            f"- instruction only: {report.instruction_only}",
            f"- completion only: {report.completion_only}",
            f"- both wrong: {report.both_wrong}",
            f"- completion correct but instruction letter wrong "
            f"(value-aligned would fix): {report.letter_wrong_value_right}",
            "",
            "Interpretation: large `completion only` with low `letter_wrong_value_right` "
            "→ model gap on MCQ interface, not parsing.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit instruction choice_only eval results.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="One or more *_examples_choice_only.json result files",
    )
    parser.add_argument(
        "--instruction",
        type=Path,
        help="Instruction results JSON for cross-format compare",
    )
    parser.add_argument(
        "--completion",
        type=Path,
        help="Completion results JSON for cross-format compare",
    )
    parser.add_argument(
        "--validate-tasks",
        type=Path,
        metavar="TASKS_JSON",
        help="Validate correct_letter vs correct_index in tasks.json",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run scoring oracle (perfect JSON → 100%% accuracy)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit tasks.json validation / oracle to first N tasks",
    )
    args = parser.parse_args()

    if args.validate_tasks:
        issues = validate_tasks_json(args.validate_tasks, max_tasks=args.max_tasks)
        if issues:
            print(f"tasks.json issues ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            n = args.max_tasks or "all"
            print(f"tasks.json OK ({n} tasks checked)")

    if args.oracle:
        ok = run_oracle_check(max_tasks=args.max_tasks or 140)
        print("Oracle scoring:", "PASS" if ok else "FAIL")

    if args.instruction and args.completion:
        if not args.instruction.is_file() or not args.completion.is_file():
            parser.error("Both --instruction and --completion must exist")
        print(_format_cross(compare_instruction_completion(args.instruction, args.completion)))

    if not args.paths:
        if not (args.validate_tasks or args.oracle or (args.instruction and args.completion)):
            parser.print_help()
            sys.exit(1)
        return

    for path in args.paths:
        if not path.is_file():
            print(f"Missing: {path}", file=sys.stderr)
            sys.exit(1)
        print(_format_report(audit_results_file(path)))


if __name__ == "__main__":
    main()
