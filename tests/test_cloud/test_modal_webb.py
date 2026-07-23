"""Modal CLI wiring for Webb suite."""

import pytest

pytest.importorskip("modal")

from ravens_numerical.cloud.modal_eval import (
    CONTAINER_TASKS_JSON,
    CONTAINER_TASKS_WEBB_JSON,
    SUPPORTED_TASK_TYPES,
    _build_eval_cli_cmd,
)


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
