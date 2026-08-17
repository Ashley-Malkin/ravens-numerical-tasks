"""Tests for 5-digit OOD task remapping."""

from __future__ import annotations

from ravens_numerical.generation.remap_5digit import remap_task, remap_value


def test_remap_value_adds_offset_to_int_leaves():
    assert remap_value(185) == 10185
    assert remap_value([1, [2, 3], None]) == [10001, [10002, 10003], None]


def test_remap_task_preserves_labels():
    task = {
        "task_type": "progression",
        "matrix": [[10, 11, 12], [20, 21, 22], [20, 21, None]],
        "answer_options": [22, 21, 23, 24],
        "correct_index": 0,
        "correct_letter": "A",
    }
    out = remap_task(task)
    assert out["correct_index"] == 0
    assert out["correct_letter"] == "A"
    assert out["task_type"] == "progression"
    assert out["answer_options"] == [10022, 10021, 10023, 10024]
    assert out["matrix"][2][2] is None
    assert out["matrix"][0][0] == 10010
