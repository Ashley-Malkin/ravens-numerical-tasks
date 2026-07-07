"""Tests for Modal → local results export helpers."""

import json
import tempfile
from pathlib import Path

from ravens_numerical.cloud.modal_results_export import (
    attach_results_data,
    export_run_results,
    local_path_from_container_results,
    public_summary,
    save_results_local,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "audit_sample_choice_only.json"
RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def test_local_path_from_container_results():
    container = (
        "/root/ravens/artifacts/runs/"
        "EleutherAI--pythia-6.9b-deduped/run1/ravens_numerical/1_examples_choice_only.json"
    )
    local = local_path_from_container_results(container, local_runs_root=RUNS_ROOT)
    assert local == (
        RUNS_ROOT
        / "EleutherAI--pythia-6.9b-deduped/run1/ravens_numerical/1_examples_choice_only.json"
    )


def test_save_results_local_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        summary = {
            "results_path": (
                "/root/ravens/artifacts/runs/"
                "stub/run1/ravens_numerical/1_examples_choice_only.json"
            ),
            "accuracy": 0.5,
            "results_data": data,
        }
        out = save_results_local(summary, local_runs_root=tmp_path)
        assert out is not None
        assert out.is_file()
        assert json.loads(out.read_text()) == data
        assert "results_data" not in summary
        assert summary["local_results_path"] == str(out)


def test_attach_and_public_summary():
    summary = {"results_path": str(FIXTURE), "accuracy": 1.0}
    enriched = attach_results_data(summary, FIXTURE)
    assert "results_data" in enriched
    assert "results_data" not in public_summary(enriched)


def test_export_run_results_both():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        payload = [{"trial": 1}]
        summary = {
            "instruction": {
                "results_path": (
                    "/root/ravens/artifacts/runs/"
                    "m/run/ravens_numerical/1_examples_choice_only.json"
                ),
                "results_data": payload,
            },
            "completion": {
                "results_path": (
                    "/root/ravens/artifacts/runs/"
                    "m/run/ravens_numerical/1_examples_completion.json"
                ),
                "results_data": payload,
            },
        }
        saved = export_run_results(summary, "both", local_runs_root=tmp_path)
        assert len(saved) == 2
