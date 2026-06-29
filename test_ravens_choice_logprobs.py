"""Tests for ravens_choice_logprobs."""

from ravens_choice_logprobs import (
    letter_logprobs_to_probs,
    multiclass_brier_score,
    parse_logprobs_by_letter,
    parse_logprobs_by_letter_vllm,
    vllm_logprobs_to_ollama_list,
)


def test_parse_logprobs_by_letter_finds_choice_token():
    resp = {
        "logprobs": [
            {"token": '{"choice":"', "logprob": -0.1, "top_logprobs": []},
            {
                "token": "B",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": "A", "logprob": -1.0},
                    {"token": "B", "logprob": -0.2},
                    {"token": "C", "logprob": -2.0},
                ],
            },
            {"token": '"}', "logprob": -0.3, "top_logprobs": []},
        ]
    }
    pred, best = parse_logprobs_by_letter(resp, '{"choice":"B"}')
    assert pred == 1
    assert best["B"] == -0.2


def test_vllm_completions_adapter():
    resp = {
        "choices": [
            {
                "text": '{"choice":"C"}',
                "logprobs": {
                    "tokens": ['{"choice":"', "C", '"}'],
                    "token_logprobs": [-0.1, -0.4, -0.2],
                    "top_logprobs": [
                        {},
                        {"A": -2.0, "C": -0.4, "D": -3.0},
                        {},
                    ],
                },
            }
        ]
    }
    entries = vllm_logprobs_to_ollama_list(resp)
    assert len(entries) == 3
    pred, _ = parse_logprobs_by_letter_vllm(resp, '{"choice":"C"}')
    assert pred == 2


def test_brier_perfect_prediction():
    probs = letter_logprobs_to_probs(
        {"A": -5.0, "B": -0.1, "C": -5.0, "D": -5.0}
    )
    assert probs is not None
    assert multiclass_brier_score(probs, 1) < 0.01
