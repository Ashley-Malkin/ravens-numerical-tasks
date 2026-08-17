"""Tests for Webb NPZ → tasks_webb.json export and suite loading."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ravens_numerical.eval.cli import Config, _default_ravens_tasks_json, _instantiate_task, resolve_n_examples, resolve_tasks
from ravens_numerical.eval.tasks.base import ModelResponse
from ravens_numerical.eval.tasks.ravens import RavensNumericalTask
from ravens_numerical.generation.export_webb import (
    export_webb_corpus,
    reduce_to_four_options,
    write_tasks_webb_json,
)
from ravens_numerical.paths import LEGACY_DATA_DIR, COMPLETE_JSON, TASKS_JSON, TASKS_WEBB_JSON
from ravens_numerical.prompts.prompts import (
    IN_CONTEXT_EXAMPLES,
    build_completion_prompt,
    build_prompt,
    matrix_to_completion_query,
)


_NPZ = LEGACY_DATA_DIR / "matrix" / "all_problems.npz"


@pytest.fixture(scope="module")
def webb_data() -> dict:
    assert TASKS_WEBB_JSON.is_file()
    return json.loads(TASKS_WEBB_JSON.read_text(encoding="utf-8"))


def test_tasks_webb_json_canonical_count(webb_data):
    assert len(webb_data["tasks"]) == 151
    assert len(webb_data["icl"]) == 31
    assert "AND_permuted" not in webb_data["icl"]


def test_tasks_webb_type_coverage(webb_data):
    counts = Counter(t["task_type"] for t in webb_data["tasks"])
    assert counts["prog_size2"] == 1
    for rt, n in counts.items():
        if rt == "prog_size2":
            continue
        assert n == 5, (rt, n)
    assert set(counts) == set(webb_data["icl"])


def test_tasks_webb_four_options_and_letters(webb_data):
    for t in webb_data["tasks"]:
        assert len(t["answer_options"]) == 4
        assert t["correct_letter"] == "ABCD"[t["correct_index"]]
        assert isinstance(t["perm_invariant"], bool)
        assert "webb_index" in t
    for rt, items in webb_data["icl"].items():
        assert len(items) == 3, rt
        for ex in items:
            assert len(ex["answer_options"]) == 4
            assert ex["correct_letter"] == "ABCD"[ex["correct_index"]]


def test_reduce_to_four_options_deterministic():
    choices = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    a1, i1, l1 = reduce_to_four_options(choices, 3, rng=random.Random(0))
    a2, i2, l2 = reduce_to_four_options(choices, 3, rng=random.Random(0))
    assert a1 == a2 and i1 == i2 and l1 == l2
    assert a1[i1] == 4
    assert len(a1) == 4


def test_export_webb_corpus_matches_checked_in(webb_data):
    fresh = export_webb_corpus(_NPZ)
    assert len(fresh["tasks"]) == len(webb_data["tasks"])
    assert fresh["tasks"][0] == webb_data["tasks"][0]
    assert fresh["icl"]["row_constant"] == webb_data["icl"]["row_constant"]


def test_write_tasks_webb_json_roundtrip(tmp_path: Path):
    out = tmp_path / "tasks_webb.json"
    write_tasks_webb_json(output=out, npz_path=_NPZ)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["tasks"]) == 151


def test_icl_disjoint_from_eval_tasks(webb_data):
    def fingerprint(item: dict) -> tuple:
        return (
            item["task_type"],
            json.dumps(item["matrix"], separators=(",", ":")),
            json.dumps(item["answer_options"], separators=(",", ":")),
            item["correct_index"],
        )

    eval_fps = {fingerprint(t) for t in webb_data["tasks"]}
    for rt, items in webb_data["icl"].items():
        for ex in items:
            assert fingerprint(ex) not in eval_fps, rt


def test_webb_suite_cli_defaults():
    cfg = Config(models=["m"], task_type="webb")
    assert resolve_tasks(cfg) == ["ravens_numerical"]
    assert resolve_n_examples(cfg) == [0, 3]
    assert _default_ravens_tasks_json(cfg) == TASKS_WEBB_JSON


def test_webb_instantiate_loads_icl_bank():
    cfg = Config(models=["m"], task_type="webb", ravens_max_tasks=2)
    task = _instantiate_task("ravens_numerical", cfg)
    assert isinstance(task, RavensNumericalTask)
    assert task._tasks_path == TASKS_WEBB_JSON
    assert task._icl_examples is not None
    assert "row_constant" in task._icl_examples


def test_ravens_instantiate_does_not_load_webb_icl():
    cfg = Config(models=["m"], task_type="ravens", ravens_max_tasks=1)
    task = _instantiate_task("ravens_numerical", cfg)
    assert task._tasks_path == COMPLETE_JSON
    assert task._icl_examples is None


def test_webb_instruction_prompt_uses_webb_icl():
    task = RavensNumericalTask(
        tasks_json=TASKS_WEBB_JSON,
        max_tasks=1,
        prompt_mode="choice_only",
        load_icl_from_json=True,
    )
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=1)
    assert "single JSON object" in prompt
    assert "Options (each has a letter A, B, C, or D)" in prompt
    # First held-out ICL for this rule type should appear.
    rule = stimulus.metadata["task"]["task_type"]
    icl0 = task._icl_examples[rule][0]
    assert format_snippet(icl0["matrix"]) in prompt


def format_snippet(matrix) -> str:
    from ravens_numerical.prompts.prompts import format_matrix

    return format_matrix(matrix).split("\n")[0]


def test_webb_completion_formats_tuple_cells():
    task = RavensNumericalTask(
        tasks_json=TASKS_WEBB_JSON,
        prompt_type="completion",
        load_icl_from_json=True,
    )
    # Find a multi-digit matrix.
    stim = None
    for s in task.canonical_stimuli():
        m = s.metadata["task"]["matrix"]
        if any(isinstance(c, list) for row in m for c in row if c is not None):
            stim = s
            break
    assert stim is not None
    prompt = task.build_prompt(stim, n_examples=0)
    assert prompt.endswith("[")
    assert " " in prompt.split("\n")[0].strip("[]") or "[" in prompt
    # Explicit Webb-style bracket serialization
    q = matrix_to_completion_query(stim.metadata["task"]["matrix"])
    assert q == prompt


def test_webb_completion_perm_invariant_scoring():
    task = RavensNumericalTask(
        tasks_json=TASKS_WEBB_JSON,
        prompt_type="completion",
        load_icl_from_json=True,
    )
    stim = next(
        s
        for s in task.canonical_stimuli()
        if s.metadata.get("perm_invariant") and " " in s.expected
    )
    tokens = stim.expected.split()
    shuffled = " ".join(reversed(tokens))
    assert task.score(ModelResponse(text=shuffled + "]"), stim) is True
    assert task.score(ModelResponse(text="999]"), stim) is False


def test_ravens_combine_is_order_variant_intersection_invariant():
    from ravens_numerical.paths import TASKS_JSON
    from ravens_numerical.prompts.prompts import completion_perm_invariant

    assert completion_perm_invariant("combine") is False
    assert completion_perm_invariant("intersection") is True

    task = RavensNumericalTask(tasks_json=TASKS_JSON, prompt_type="completion")
    combine = next(
        s for s in task.canonical_stimuli() if s.metadata["task"]["task_type"] == "combine"
    )
    inter = next(
        s
        for s in task.canonical_stimuli()
        if s.metadata["task"]["task_type"] == "intersection"
    )
    assert combine.metadata.get("perm_invariant") is False
    assert inter.metadata.get("perm_invariant") is True

    c_tokens = combine.expected.split()
    assert len(c_tokens) >= 2
    c_shuffled = " ".join(reversed(c_tokens))
    if c_shuffled != combine.expected:
        assert task.score(ModelResponse(text=c_shuffled + "]"), combine) is False
    assert task.score(ModelResponse(text=combine.expected + "]"), combine) is True
    assert task.score(ModelResponse(text=inter.expected + "]"), inter) is True


def test_webb_icl_missing_type_raises():
    fake_task = {
        "task_type": "not_a_real_webb_type",
        "matrix": [[1, 1, 1], [1, 1, 1], [1, 1, None]],
        "answer_options": [1, 2, 3, 4],
        "correct_index": 0,
        "correct_letter": "A",
    }
    with pytest.raises(KeyError, match="No ICL examples"):
        build_prompt(fake_task, mode="choice_only", n_examples=1, icl_examples={"row_constant": []})


def test_ravens_prompts_unchanged_without_icl_bank():
    # Default Raven bank still used; unknown types still fall back to constancy.
    task = {
        "task_type": "totally_unknown",
        "matrix": [[1, 1, 1], [1, 1, 1], [1, 1, None]],
        "answer_options": [1, 2, 3, 4],
        "correct_index": 0,
        "correct_letter": "A",
    }
    prompt = build_prompt(task, mode="plain", n_examples=1)
    # Constancy ICL demo appears (11s matrix from first constancy example).
    assert "11 | 11 | 11" in prompt
    assert len(IN_CONTEXT_EXAMPLES["constancy"]) == 3


def test_unknown_custom_icl_does_not_use_raven_bank():
    fake_task = {
        "task_type": "constancy",
        "matrix": [[1, 1, 1], [1, 1, 1], [1, 1, None]],
        "answer_options": [1, 2, 3, 4],
        "correct_index": 0,
        "correct_letter": "A",
    }
    with pytest.raises(KeyError):
        build_completion_prompt(fake_task, n_examples=1, icl_examples={"row_constant": IN_CONTEXT_EXAMPLES["constancy"]})
