#!/usr/bin/env python3
"""Tests for ravens_eval_models registry helpers."""

from __future__ import annotations

import pytest

from ravens_numerical.models.registry import (
    BABYLM_SCALING_MODELS,
    MINIBERTA_SCALING_MODELS,
    OLMO2_CHECKPOINT_REVISIONS_BY_MODEL,
    OLMO2_CHECKPOINT_SCALING_MODELS,
    OLMO2_CHECKPOINT_TOKEN_BUDGETS_B,
    OLMO2_MODELS,
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    PYTHIA_CHECKPOINT_STEPS,
    PYTHIA_SCALING_MODELS,
    QWEN3_8B_BASE,
    QWEN3_SCALING_MODELS,
    base_model_id,
    checkpoint_step_from_model_id,
    format_olmo2_checkpoint_model_id,
    format_pythia_checkpoint_model_id,
    gpu_tier_for_model,
    is_babylm_model,
    is_miniberta_model,
    is_olmo2_base_model,
    is_olmo2_checkpoint_model_id,
    is_olmo2_instruct_model,
    is_olmo2_model,
    is_pythia_checkpoint_model_id,
    is_qwen3_base_model,
    is_qwen3_instruct_model,
    is_sft_checkpoint_model,
    is_sft_run_id,
    max_model_len_for_model,
    model_family,
    olmo2_checkpoint_model_ids,
    olmo2_tokens_billions_from_revision,
    parse_miniberta_seed,
    parse_params_billions,
    parse_pythia_checkpoint_model_id,
    parse_training_corpus_millions,
    pythia_checkpoint_model_ids,
    resolve_models_arg,
    resolve_sft_model_id,
    sft_run_id_from_model_id,
    uses_completions_choice_only,
)
from ravens_numerical.paths import (
    BABYLM_EXPERIMENTS_MD,
    BABYLM_FINETUNE_LOGS_DIR,
    babylm_sft_experiments_md,
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
    assert (
        uses_completions_choice_only("EleutherAI/pythia-70m-deduped@step64") is True
    )
    assert uses_completions_choice_only(QWEN3_8B_BASE) is True
    assert uses_completions_choice_only("Qwen/Qwen3-8B") is False
    assert uses_completions_choice_only(MINIBERTA_1M) is False


def test_pythia_checkpoint_model_ids():
    ids = pythia_checkpoint_model_ids()
    assert len(ids) == len(PYTHIA_CHECKPOINT_SCALING_MODELS) * len(
        PYTHIA_CHECKPOINT_STEPS
    )
    assert ids[0] == format_pythia_checkpoint_model_id(
        PYTHIA_CHECKPOINT_SCALING_MODELS[0], PYTHIA_CHECKPOINT_STEPS[0]
    )
    assert resolve_models_arg("pythia-checkpoints") == ids


def test_parse_pythia_checkpoint_model_id():
    base, rev = parse_pythia_checkpoint_model_id(
        "EleutherAI/pythia-70m-deduped@step25000"
    )
    assert base == "EleutherAI/pythia-70m-deduped"
    assert rev == "step25000"
    assert base_model_id("EleutherAI/pythia-70m-deduped@step64") == (
        "EleutherAI/pythia-70m-deduped"
    )
    assert is_pythia_checkpoint_model_id(
        "EleutherAI/pythia-70m-deduped@step64"
    )
    assert is_pythia_checkpoint_model_id("EleutherAI/pythia-70m-deduped") is False


def test_checkpoint_step_from_model_id():
    assert (
        checkpoint_step_from_model_id("EleutherAI/pythia-70m-deduped@step64")
        == 64
    )
    assert (
        checkpoint_step_from_model_id(
            "allenai/OLMo2-7B-1124@stage1-step5000-tokens21B"
        )
        == 5000
    )
    assert checkpoint_step_from_model_id("EleutherAI/pythia-70m-deduped") is None
    assert checkpoint_step_from_model_id("Qwen/Qwen3-8B") is None


def test_olmo2_model_family():
    assert is_olmo2_model("allenai/OLMo2-7B-1124") is True
    assert is_olmo2_base_model("allenai/OLMo2-7B-1124") is True
    assert is_olmo2_instruct_model("allenai/OLMo-2-1124-7B-Instruct") is True
    assert is_olmo2_base_model("allenai/OLMo-2-1124-7B-Instruct") is False
    assert uses_completions_choice_only("allenai/OLMo2-7B-1124") is True
    assert uses_completions_choice_only("allenai/OLMo-2-1124-7B-Instruct") is False


def test_gpu_tier_for_pythia_checkpoint():
    assert gpu_tier_for_model("EleutherAI/pythia-12b-deduped@step1000") == "A100"
    assert gpu_tier_for_model("EleutherAI/pythia-70m-deduped@step64") == "T4"


def test_resolve_models_arg_olmo2():
    assert resolve_models_arg("olmo2") == list(OLMO2_MODELS)
    assert resolve_models_arg("olmo") == list(OLMO2_MODELS)


def test_resolve_models_arg_rejects_bare_unknown():
    with pytest.raises(ValueError, match="Unknown model id 'olmoo'"):
        resolve_models_arg("olmoo")


def test_olmo2_checkpoint_model_ids():
    assert OLMO2_CHECKPOINT_TOKEN_BUDGETS_B == (1, 21, 42, 49, 63)
    for base in OLMO2_CHECKPOINT_SCALING_MODELS:
        revs = OLMO2_CHECKPOINT_REVISIONS_BY_MODEL[base]
        assert len(revs) >= len(OLMO2_CHECKPOINT_TOKEN_BUDGETS_B)
    ids = olmo2_checkpoint_model_ids()
    assert len(ids) == sum(
        len(OLMO2_CHECKPOINT_REVISIONS_BY_MODEL[base])
        for base in OLMO2_CHECKPOINT_SCALING_MODELS
    )
    assert (
        len(OLMO2_CHECKPOINT_REVISIONS_BY_MODEL["allenai/OLMo2-7B-1124"]) == 7
    )
    assert (
        len(OLMO2_CHECKPOINT_REVISIONS_BY_MODEL["allenai/OLMo-2-13B-1124"])
        == 6
    )
    assert ids[0] == format_olmo2_checkpoint_model_id(
        OLMO2_CHECKPOINT_SCALING_MODELS[0],
        OLMO2_CHECKPOINT_REVISIONS_BY_MODEL[OLMO2_CHECKPOINT_SCALING_MODELS[0]][0],
    )
    assert resolve_models_arg("olmo2-checkpoints") == ids
    assert all("@" in mid for mid in ids)


def test_parse_olmo2_checkpoint_model_id():
    mid = "allenai/OLMo2-7B-1124@stage1-step5000-tokens21B"
    base, rev = parse_pythia_checkpoint_model_id(mid)
    assert base == "allenai/OLMo2-7B-1124"
    assert rev == "stage1-step5000-tokens21B"
    assert base_model_id(mid) == "allenai/OLMo2-7B-1124"
    assert is_olmo2_checkpoint_model_id(mid) is True
    assert is_pythia_checkpoint_model_id(mid) is False
    assert is_olmo2_checkpoint_model_id("allenai/OLMo2-7B-1124") is False
    assert olmo2_tokens_billions_from_revision(rev) == 21.0
    assert gpu_tier_for_model(mid) == "A10G"
    assert gpu_tier_for_model(
        "allenai/OLMo-2-13B-1124@stage1-step6000-tokens51B"
    ) == "A100"


SFT_RUN_ID = "babylm-10m-gpt2__all_types__20260723T212815Z"
SFT_CHECKPOINT_PATH = f"/checkpoints/{SFT_RUN_ID}"


def test_sft_run_id_detection_and_resolve():
    assert is_sft_run_id(SFT_RUN_ID) is True
    assert is_sft_run_id(BABYLM_10M) is False
    assert is_sft_run_id("olmoo") is False
    assert is_sft_checkpoint_model(SFT_RUN_ID) is True
    assert is_sft_checkpoint_model(SFT_CHECKPOINT_PATH) is True
    assert is_sft_checkpoint_model(BABYLM_10M) is False
    assert resolve_sft_model_id(SFT_RUN_ID) == SFT_CHECKPOINT_PATH
    assert resolve_sft_model_id(SFT_CHECKPOINT_PATH) == SFT_CHECKPOINT_PATH
    assert sft_run_id_from_model_id(SFT_RUN_ID) == SFT_RUN_ID
    assert sft_run_id_from_model_id(SFT_CHECKPOINT_PATH) == SFT_RUN_ID


def test_resolve_models_arg_sft_run_id():
    assert resolve_models_arg(SFT_RUN_ID) == [SFT_CHECKPOINT_PATH]
    assert resolve_models_arg(SFT_CHECKPOINT_PATH) == [SFT_CHECKPOINT_PATH]
    assert gpu_tier_for_model(SFT_RUN_ID) == "T4"
    assert gpu_tier_for_model(SFT_CHECKPOINT_PATH) == "T4"


def test_babylm_sft_experiments_log_path():
    sft_log = babylm_sft_experiments_md(SFT_RUN_ID)
    assert sft_log == BABYLM_FINETUNE_LOGS_DIR / f"{SFT_RUN_ID}.md"
    assert babylm_sft_experiments_md(SFT_CHECKPOINT_PATH) == sft_log
    assert (
        babylm_sft_experiments_md(SFT_RUN_ID, n_examples=0)
        == BABYLM_FINETUNE_LOGS_DIR / f"{SFT_RUN_ID}__n0.md"
    )
    assert (
        babylm_sft_experiments_md(SFT_RUN_ID, n_examples=3)
        == BABYLM_FINETUNE_LOGS_DIR / f"{SFT_RUN_ID}__n3.md"
    )
    # Hub BabyLM baselines keep the shared artifacts log, not per-run SFT logs.
    assert BABYLM_EXPERIMENTS_MD.name == "babyLMexperiments.md"
    assert BABYLM_EXPERIMENTS_MD != sft_log


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
    test_pythia_checkpoint_model_ids()
    test_parse_pythia_checkpoint_model_id()
    test_checkpoint_step_from_model_id()
    test_olmo2_model_family()
    test_gpu_tier_for_pythia_checkpoint()
    test_resolve_models_arg_olmo2()
    test_resolve_models_arg_rejects_bare_unknown()
    test_olmo2_checkpoint_model_ids()
    test_parse_olmo2_checkpoint_model_id()
    test_sft_run_id_detection_and_resolve()
    test_resolve_models_arg_sft_run_id()
    test_babylm_sft_experiments_log_path()
    print("OK")
