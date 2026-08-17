"""Tests for challenge task generation."""

from __future__ import annotations

from collections import Counter

from ravens_numerical.generation.challenge import (
    PROGRESSION_STEPS,
    build_challenge_tasks,
    generate_distribution_of_three_task,
    generate_progression_plus_n_task,
    generate_tuple_grid_task,
)
from ravens_numerical.prompts.prompts import (
    expected_completion_answer,
    format_completion_answer,
)


def test_distribution_of_three_distractors_from_triple():
    task = generate_distribution_of_three_task(rng=__import__("random").Random(0))
    row0 = task["matrix"][0]
    triple = set(row0)
    correct = task["answer_options"][task["correct_index"]]
    assert correct in triple
    distractors = [o for o in task["answer_options"] if o != correct]
    from_triple = [d for d in distractors if d in triple]
    assert len(from_triple) == 2
    assert sum(1 for d in distractors if d not in triple) == 1


def test_progression_plus_n_matches_step():
    task = generate_progression_plus_n_task(step=5, rng=__import__("random").Random(1))
    assert task["step"] == 5
    m = task["matrix"]
    for r in range(2):
        assert m[r][1] - m[r][0] == 5
        assert m[r][2] - m[r][1] == 5
    expect = m[2][0] + 10
    assert task["answer_options"][task["correct_index"]] == expect


def test_tuple_grid_order_variant_options():
    task = generate_tuple_grid_task(rng=__import__("random").Random(2))
    m = task["matrix"]
    row_keys = [m[i][0][0] for i in range(3)]
    col_keys = [m[0][j][1] for j in range(3)]
    correct = [row_keys[2], col_keys[2]]
    swapped = [col_keys[2], row_keys[2]]
    opts = task["answer_options"]
    assert opts[task["correct_index"]] == correct
    assert row_keys[2] in opts
    assert swapped in opts
    assert task.get("perm_invariant") is False
    # Completion formatting distinguishes order.
    assert format_completion_answer(correct) != format_completion_answer(swapped)
    assert expected_completion_answer(task) == format_completion_answer(correct)


def test_build_challenge_tasks_counts_and_steps():
    tasks = build_challenge_tasks(n_per_type=10, seed=99)
    counts = Counter(t["task_type"] for t in tasks)
    assert counts == {
        "distribution_of_three": 10,
        "progression_plus_n": 10,
        "tuple_grid": 10,
    }
    assert [t["task_type"] for t in tasks[:3]] == [
        "distribution_of_three",
        "progression_plus_n",
        "tuple_grid",
    ]
    steps = Counter(
        t["step"] for t in tasks if t["task_type"] == "progression_plus_n"
    )
    assert set(steps) == set(PROGRESSION_STEPS)
    assert sum(steps.values()) == 10
