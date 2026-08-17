"""Modal CLI wiring for Webb suite + SFT experiment log routing."""

import pytest

pytest.importorskip("modal")

from ravens_numerical.cloud.modal_eval import (
    CONTAINER_TASKS_JSON,
    CONTAINER_TASKS_WEBB_JSON,
    SUPPORTED_TASK_TYPES,
    _build_eval_cli_cmd,
    _experiments_path_for_model,
    resolve_ravens_tasks_json_arg,
)
from ravens_numerical.paths import (
    BABYLM_EXPERIMENTS_MD,
    BABYLM_FINETUNE_LOGS_DIR,
)
from pathlib import Path


def test_supported_task_types_include_webb():
    assert "webb" in SUPPORTED_TASK_TYPES


def test_build_eval_cli_cmd_webb_uses_tasks_webb_json():
    cmd = _build_eval_cli_cmd(
        backend="vllm",
        model="EleutherAI/pythia-70m-deduped",
        n_examples=0,
        task_type="webb",
        prompt_type="instruction",
        ravens_prompt_mode="choice_only",
        max_tasks=10,
        base_url="http://127.0.0.1:8000",
    )
    assert "--task-type" in cmd
    assert "webb" in cmd
    assert CONTAINER_TASKS_WEBB_JSON in cmd
    assert CONTAINER_TASKS_JSON not in cmd


def test_build_eval_cli_cmd_ravens_uses_tasks_json():
    cmd = _build_eval_cli_cmd(
        backend="vllm",
        model="EleutherAI/pythia-70m-deduped",
        n_examples=0,
        task_type="ravens",
        prompt_type="completion",
        ravens_prompt_mode="auto",
        max_tasks=None,
        base_url="http://127.0.0.1:8000",
    )
    assert CONTAINER_TASKS_JSON in cmd
    assert CONTAINER_TASKS_WEBB_JSON not in cmd


def test_build_eval_cli_cmd_test_data_override():
    override = "/root/ravens/babylm_finetune/data/oneICL_test.json"
    cmd = _build_eval_cli_cmd(
        backend="vllm",
        model="EleutherAI/pythia-70m-deduped",
        n_examples=1,
        task_type="ravens",
        prompt_type="completion",
        ravens_prompt_mode="auto",
        max_tasks=None,
        base_url="http://127.0.0.1:8000",
        ravens_tasks_json=override,
    )
    assert override in cmd
    assert CONTAINER_TASKS_JSON not in cmd


def test_ravens_tasks_json_selected_by_n_examples():
    from ravens_numerical.cloud.modal_eval import ravens_tasks_json_for_n_examples

    assert ravens_tasks_json_for_n_examples(0, "ravens") is None
    assert ravens_tasks_json_for_n_examples(1, "webb") is None
    one = ravens_tasks_json_for_n_examples(1, "ravens")
    three = ravens_tasks_json_for_n_examples(3, "ravens")
    assert one is not None and one.endswith("oneICL_test.json")
    assert three is not None and three.endswith("threeICL_test.json")


def test_ravens_tasks_json_aliases_complete_default_and_tasks_backup():
    from ravens_numerical.paths import COMPLETE_JSON, TASKS_JSON

    assert resolve_ravens_tasks_json_arg("default").resolve() == COMPLETE_JSON.resolve()
    assert resolve_ravens_tasks_json_arg("complete").resolve() == COMPLETE_JSON.resolve()
    assert resolve_ravens_tasks_json_arg("tasks").resolve() == TASKS_JSON.resolve()


def test_experiments_path_sft_vs_hub_babylm():
    run_id = "babylm-10m-gpt2__all_types__20260723T212815Z"
    sft_path = _experiments_path_for_model(run_id, n_examples=0)
    assert sft_path == BABYLM_FINETUNE_LOGS_DIR / f"{run_id}__n0.md"
    assert (
        _experiments_path_for_model(f"/checkpoints/{run_id}", n_examples=1)
        == BABYLM_FINETUNE_LOGS_DIR / f"{run_id}__n1.md"
    )
    assert (
        _experiments_path_for_model("BabyLM-community/babylm-baseline-10m-gpt2")
        == BABYLM_EXPERIMENTS_MD
    )


def test_combined_sft_log_path_encodes_n_examples():
    from ravens_numerical.paths import combined_sft_log_path_for_n_examples

    assert (
        combined_sft_log_path_for_n_examples(None, 0).name == "all_sft_evals_n0.md"
    )
    assert (
        combined_sft_log_path_for_n_examples(
            Path("babylm_finetune/logs/newsplitSFT.md"), 1
        ).name
        == "newsplitSFT_n1.md"
    )
    # Already suffixed paths are left alone.
    assert (
        combined_sft_log_path_for_n_examples(
            Path("babylm_finetune/logs/newsplitSFT_n3.md"), 1
        ).name
        == "newsplitSFT_n3.md"
    )
