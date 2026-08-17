"""Unit tests for echo completion logprob helpers."""

from __future__ import annotations

import pytest

from ravens_numerical.scoring.echo_logprobs import (
    sum_echo_completion_logprobs,
    unique_logprob_argmax,
)


def test_sum_includes_straddling_token():
    # prompt_len=2 ("x["), full "x[185]" len 6; token at 1 covers [1,6)
    assert sum_echo_completion_logprobs(
        [None, -1.25],
        [0, 1],
        2,
        full_len=6,
    ) == pytest.approx(-1.25)


def test_sum_returns_none_when_empty_overlap():
    # Tokens only cover the prompt; completion-span token has lp None.
    assert (
        sum_echo_completion_logprobs(
            [None, -1.0, None],
            [0, 3, 6],
            prompt_len=6,
            full_len=8,
        )
        is None
    )


def test_sum_excludes_prompt_only_tokens():
    assert sum_echo_completion_logprobs(
        [None, -5.0, -1.0, -2.0],
        [0, 3, 6, 9],
        prompt_len=6,
        full_len=11,
    ) == pytest.approx(-3.0)


def test_unique_logprob_argmax_picks_unique_max():
    assert unique_logprob_argmax({"a": -3.0, "b": -1.0, "c": -2.0}) == "b"


def test_unique_logprob_argmax_none_on_tie():
    assert unique_logprob_argmax({"a": 0.0, "b": 0.0, "c": 0.0}) is None
    assert unique_logprob_argmax({"a": -1.0, "b": -1.0}) is None


def test_unique_logprob_argmax_none_when_empty():
    assert unique_logprob_argmax({}) is None


def test_unique_logprob_argmax_none_on_collapsed_near_zero():
    """Float noise among ≈0 sums must not produce a unique winner."""
    assert (
        unique_logprob_argmax(
            {"300": -1e-6, "301": -2e-6, "302": -3e-7, "299": -4e-6}
        )
        is None
    )


def test_unique_logprob_argmax_length_normalizes():
    """Sum prefers the short option; mean per token prefers the pair."""
    logprobs = {"10": -0.10, "10 11": -0.12, "11 10": -3.0, "7 10": -3.0}
    assert unique_logprob_argmax(logprobs, length_normalize=False) == "10"
    assert unique_logprob_argmax(logprobs) == "10 11"


def test_completion_option_length():
    from ravens_numerical.scoring.echo_logprobs import completion_option_length

    assert completion_option_length("10") == 1
    assert completion_option_length("10]") == 1
    assert completion_option_length("10 11]") == 2
