"""Tests for audit_instruction_results."""

from pathlib import Path

from ravens_numerical.analysis.audit import (
    audit_results_file,
    compare_instruction_completion,
    run_oracle_check,
    validate_tasks_json,
)
from ravens_numerical.paths import TASKS_JSON

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "audit_sample_choice_only.json"


def test_validate_tasks_json_no_issues():
    issues = validate_tasks_json(TASKS_JSON, max_tasks=140)
    assert issues == []


def test_audit_fixture_sample():
    report = audit_results_file(FIXTURE)
    assert report.total == 3
    assert report.text_correct == 1
    assert report.logprob_correct == 1
    assert report.logprob_available == 2
    assert report.parse_none == 1
    assert report.stored_mismatch == 0
    assert report.text_logprob_disagree == 0


def test_oracle_scoring():
    assert run_oracle_check(max_tasks=5) is True


def test_compare_instruction_completion_requires_overlap():
    report = compare_instruction_completion(FIXTURE, FIXTURE)
    assert report.matched == 1
    assert report.both_wrong == 2
