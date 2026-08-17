"""Tests for experiment markdown log formatting."""

from __future__ import annotations

import re

from ravens_numerical.analysis.experiment_log import (
    format_all_sft_evals_md,
    format_experiment_entry,
)

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
        n_examples=0,
    )
    header = entry.splitlines()[0]
    assert re.match(
        r"^## \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC — run_ravens_eval \(pythia, aba\) "
        r"\(max_tasks=14\) \(n_examples=0\)$",
        header,
    )
    assert "(step=" not in header
    assert "`--n-examples 0`" in entry


def test_format_experiment_entry_includes_step_when_provided():
    entry = format_experiment_entry(
        run_label="run_ravens_eval (pythia@step64, aba)",
        model_summaries={"pythia": _SUMMARY},
        max_tasks=14,
        settings_note="test settings",
        step=64,
        n_examples=3,
    )
    header = entry.splitlines()[0]
    assert header.endswith("(max_tasks=14) (n_examples=3) (step=64)")
    # Keep (max_tasks=N) parseable as its own parenthetical.
    assert re.search(r"\(max_tasks=14\)", header)
    assert re.search(r"\(n_examples=3\)", header)


def test_format_all_sft_evals_md_puts_run_id_before_scores():
    rid_a = "babylm-10m-gpt2__all_types__20260723T220459Z"
    rid_b = "babylm-100m-gpt2__all_types__20260723T222637Z"
    text = format_all_sft_evals_md(
        results_by_run_id={
            rid_b: {
                "accuracy": 0.8,
                "correct": 280,
                "total": 350,
                "by_task_type": {
                    "constancy": {"accuracy": 1.0, "correct": 50, "total": 50},
                },
            },
            rid_a: {
                "accuracy": 0.7,
                "correct": 245,
                "total": 350,
                "by_task_type": {
                    "progression": {"accuracy": 0.4, "correct": 20, "total": 50},
                },
            },
        },
        settings_note="test settings",
        max_tasks=None,
        n_examples=1,
    )
    assert "## `" + rid_a + "`" in text
    assert "## `" + rid_b + "`" in text
    # Sorted by run_id (lexicographic: 100m before 10m).
    pos_a = text.index(f"## `{rid_a}`")
    pos_b = text.index(f"## `{rid_b}`")
    assert pos_b < pos_a
    assert text.index("**Overall:** 80.0%", pos_b) < pos_a
    assert "**By task:**" in text
    assert "all tasks" in text
    assert "n_examples=1" in text
    assert "`--n-examples 1`" in text
