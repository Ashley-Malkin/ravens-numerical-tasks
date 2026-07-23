"""Tests for experiment markdown log formatting."""

from __future__ import annotations

import re

from ravens_numerical.analysis.experiment_log import format_experiment_entry

_SUMMARY = {
    "model_id": "EleutherAI/pythia-70m-deduped@step64",
    "accuracy": 0.5,
    "correct": 7,
    "total": 14,
    "by_task_type": {
        "combine": {"accuracy": 0.5, "correct": 1, "total": 2},
    },
}


def test_format_experiment_entry_includes_run_time_in_header():
    entry = format_experiment_entry(
        run_label="run_ravens_eval (pythia, aba)",
        model_summaries={"pythia": _SUMMARY},
        max_tasks=14,
        settings_note="test settings",
    )
    header = entry.splitlines()[0]
    assert re.match(
        r"^## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC — run_ravens_eval \(pythia, aba\) \(max_tasks=14\)$",
        header,
    )
    assert "(step=" not in header


def test_format_experiment_entry_includes_step_when_provided():
    entry = format_experiment_entry(
        run_label="run_ravens_eval (pythia@step64, aba)",
        model_summaries={"pythia": _SUMMARY},
        max_tasks=14,
        settings_note="test settings",
        step=64,
    )
    header = entry.splitlines()[0]
    assert header.endswith("(max_tasks=14) (step=64)")
    # Keep (max_tasks=N) parseable as its own parenthetical.
    assert re.search(r"\(max_tasks=14\)", header)
