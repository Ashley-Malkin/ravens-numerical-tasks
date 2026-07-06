#!/usr/bin/env python3
"""Tests for ravens_eval_models registry helpers."""

from __future__ import annotations

from ravens_eval_models import (
    BABYLM_SCALING_MODELS,
    MINIBERTA_SCALING_MODELS,
    QWEN3_8B_BASE,
    QWEN3_SCALING_MODELS,
    is_babylm_model,
    is_miniberta_model,
    is_qwen3_base_model,
    is_qwen3_instruct_model,
    max_model_len_for_model,
    model_family,
    parse_miniberta_seed,
    parse_params_billions,
    parse_training_corpus_millions,
    resolve_models_arg,
    uses_completions_choice_only,
)

BABYLM_10M = "BabyLM-community/babylm-baseline-10m-gpt2"
BABYLM_100M = "BabyLM-community/babylm-baseline-100m-gpt2"
MINIBERTA_1M = "nyu-mll/roberta-med-small-1M-1"
MINIBERTA_100M = "nyu-mll/roberta-base-100M-2"
MINIBERTA_1B = "nyu-mll/roberta-base-1B-3"


def test_babylm_model_family():
    assert model_family(BABYLM_10M) == "babylm"
    assert model_family(BABYLM_100M) == "babylm"
    assert is_babylm_model(BABYLM_10M) is True


def test_miniberta_model_family():
    assert model_family(MINIBERTA_1M) == "miniberta"
    assert model_family(MINIBERTA_100M) == "miniberta"
    assert is_miniberta_model(MINIBERTA_1B) is True
    assert is_miniberta_model("EleutherAI/pythia-70m-deduped") is False


def test_babylm_params_and_corpus():
    assert parse_params_billions(BABYLM_10M) == 0.124
    assert parse_params_billions(BABYLM_100M) == 0.124
    assert parse_training_corpus_millions(BABYLM_10M) == 10.0
    assert parse_training_corpus_millions(BABYLM_100M) == 100.0


def test_miniberta_params_corpus_seed():
    assert parse_params_billions(MINIBERTA_1M) == 0.045
    assert parse_params_billions(MINIBERTA_100M) == 0.125
    assert parse_training_corpus_millions(MINIBERTA_1M) == 1.0
    assert parse_training_corpus_millions(MINIBERTA_100M) == 100.0
    assert parse_training_corpus_millions(MINIBERTA_1B) == 1000.0
    assert parse_miniberta_seed(MINIBERTA_100M) == 2
    assert parse_miniberta_seed(MINIBERTA_1B) == 3


def test_resolve_models_arg_babylm():
    assert resolve_models_arg("babylm") == list(BABYLM_SCALING_MODELS)


def test_resolve_models_arg_miniberta():
    resolved = resolve_models_arg("miniberta")
    assert resolved == list(MINIBERTA_SCALING_MODELS)
    assert len(resolved) == 12


def test_qwen3_base_vs_instruct():
    assert is_qwen3_base_model(QWEN3_8B_BASE) is True
    assert is_qwen3_instruct_model(QWEN3_8B_BASE) is False
    assert is_qwen3_instruct_model("Qwen/Qwen3-8B") is True
    assert is_qwen3_base_model("Qwen/Qwen3-8B") is False
    assert parse_params_billions(QWEN3_8B_BASE) == 8.0


def test_resolve_models_arg_qwen3():
    resolved = resolve_models_arg("qwen3")
    assert resolved == list(QWEN3_SCALING_MODELS)
    assert len(resolved) == 6
    assert QWEN3_8B_BASE in resolved


def test_max_model_len_babylm():
    assert max_model_len_for_model(BABYLM_10M) == 1024
    assert max_model_len_for_model(BABYLM_100M) == 1024
    assert max_model_len_for_model("EleutherAI/pythia-70m-deduped") == 2048


def test_max_model_len_miniberta():
    assert max_model_len_for_model(MINIBERTA_1M) == 512


def test_uses_completions_choice_only():
    assert uses_completions_choice_only(BABYLM_10M) is True
    assert uses_completions_choice_only("EleutherAI/pythia-70m-deduped") is True
    assert uses_completions_choice_only(QWEN3_8B_BASE) is True
    assert uses_completions_choice_only("Qwen/Qwen3-8B") is False
    assert uses_completions_choice_only(MINIBERTA_1M) is False


if __name__ == "__main__":
    test_babylm_model_family()
    test_miniberta_model_family()
    test_babylm_params_and_corpus()
    test_miniberta_params_corpus_seed()
    test_resolve_models_arg_babylm()
    test_resolve_models_arg_miniberta()
    test_qwen3_base_vs_instruct()
    test_resolve_models_arg_qwen3()
    test_max_model_len_babylm()
    test_max_model_len_miniberta()
    test_uses_completions_choice_only()
    print("OK")
