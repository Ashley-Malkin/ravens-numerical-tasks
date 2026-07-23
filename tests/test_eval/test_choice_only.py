"""Tests for instruction choice-only wiring."""

import json

import pytest
import responses as resp

from ravens_numerical.eval.cli import Config, _make_backend, _results_suffix
from ravens_numerical.eval.backends import (
    ForcedChoiceVLLMBackend,
    Olmo2ChoiceOnlyVLLMBackend,
    Olmo2InstructChoiceOnlyVLLMBackend,
    PythiaChoiceOnlyVLLMBackend,
    Qwen3ChoiceOnlyVLLMBackend,
    RobertaMLMBackend,
    strip_qwen_thinking,
)
from ravens_numerical.prompts.prompts import adapt_choice_only_prompt_for_letter_output
from ravens_numerical.eval.runner import evaluate
from ravens_numerical.eval.tasks.base import ModelBackend, ModelResponse
from ravens_numerical.eval.tasks.ravens import RavensNumericalTask


def test_ravens_task_uses_choice_only_prompt_mode():
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    assert task.uses_choice_only_metrics is True
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=0)
    assert "single JSON object" in prompt
    assert '"choice"' in prompt


def test_adapt_choice_only_prompt_for_letter_output():
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=1)
    assert "single JSON object" in prompt
    adapted = adapt_choice_only_prompt_for_letter_output(prompt)
    assert "single JSON object" not in adapted
    assert "Reply with ONLY the option letter" in adapted
    assert 'A valid reply would be exactly: {"choice"' not in adapted


def test_ravens_completion_ignores_prompt_mode():
    task = RavensNumericalTask(
        max_tasks=1,
        prompt_type="completion",
        prompt_mode="choice_only",
    )
    assert task.uses_choice_only_metrics is False
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=0)
    assert "JSON object" not in prompt
    assert prompt.endswith("[")


def test_results_suffix_choice_only():
    cfg = Config(
        models=["EleutherAI/pythia-70m-deduped"],
        tasks=["ravens_numerical"],
        ravens_prompt_type="instruction",
        ravens_prompt_mode="auto",
        score_mode="free_gen",
    )
    assert _results_suffix(cfg, "ravens_numerical") == "choice_only"


def test_make_backend_selects_family():
    cfg = Config(
        models=["EleutherAI/pythia-410m-deduped"],
        tasks=["ravens_numerical"],
        ravens_prompt_type="instruction",
        ravens_prompt_mode="choice_only",
        score_mode="free_gen",
    )
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    assert isinstance(_make_backend(cfg, cfg.models[0], task), PythiaChoiceOnlyVLLMBackend)

    cfg.models = ["Qwen/Qwen3-8B"]
    assert isinstance(_make_backend(cfg, cfg.models[0], task), Qwen3ChoiceOnlyVLLMBackend)

    cfg.models = ["Qwen/Qwen3-8B-Base"]
    assert isinstance(_make_backend(cfg, cfg.models[0], task), PythiaChoiceOnlyVLLMBackend)

    cfg.models = ["BabyLM-community/babylm-baseline-10m-gpt2"]
    assert isinstance(_make_backend(cfg, cfg.models[0], task), PythiaChoiceOnlyVLLMBackend)

    cfg.models = ["allenai/OLMo2-7B-1124"]
    assert isinstance(_make_backend(cfg, cfg.models[0], task), Olmo2ChoiceOnlyVLLMBackend)

    cfg.models = ["allenai/OLMo-2-1124-7B-Instruct"]
    assert isinstance(
        _make_backend(cfg, cfg.models[0], task), Olmo2InstructChoiceOnlyVLLMBackend
    )

    cfg.models = ["nyu-mll/roberta-base-100M-1"]
    cfg.backend = "vllm"
    assert isinstance(_make_backend(cfg, cfg.models[0], task), RobertaMLMBackend)

    cfg.backend = "hf"
    assert isinstance(_make_backend(cfg, cfg.models[0], task), RobertaMLMBackend)


def test_strip_qwen_thinking():
    open_tag = "<" + "think" + ">"
    close_tag = "</" + "think" + ">"
    text = f"{open_tag}reasoning here{close_tag}3]"
    assert strip_qwen_thinking(text) == "3]"


@resp.activate
def test_pythia_choice_only_backend_posts_guided_json():
    backend = PythiaChoiceOnlyVLLMBackend(
        "EleutherAI/pythia-410m-deduped", "http://localhost:8000"
    )
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/completions",
        json={"choices": [{"text": '{"choice":"A"}', "logprobs": {}}]},
        status=200,
    )
    result = backend.generate("prompt")
    assert result.text == '{"choice":"A"}'
    body = json.loads(resp.calls[0].request.body)
    assert "extra_body" not in body
    assert body["structured_outputs"]["json"]["required"] == ["choice"]


@resp.activate
def test_forced_choice_backend_posts_guided_choice():
    from ravens_numerical.eval.tasks.base import Stimulus
    from ravens_numerical.eval.tasks.legacy.rules import RulesTask

    backend = ForcedChoiceVLLMBackend(
        "EleutherAI/pythia-160m-deduped", "http://localhost:8000"
    )
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/completions",
        json={"choices": [{"text": " pe", "logprobs": {}}]},
        status=200,
    )
    stimulus = Stimulus(
        query="pe bi",
        expected="pe",
        answer_choices=["pe", "bi"],
        metadata={"rule": "ABA"},
    )
    result = backend.generate("pe bi", stimulus=stimulus, task=RulesTask())
    assert result.text == " pe"
    body = json.loads(resp.calls[0].request.body)
    assert "extra_body" not in body
    assert body["structured_outputs"]["choice"] == [" pe", " bi"]
    assert body["max_tokens"] == 16


def test_make_backend_forced_choice_aba_uses_plain_vllm():
    """ABA syllable options like ``" pe"`` 500 under vLLM structured choice."""
    from ravens_numerical.eval.backends import VLLMBackend
    from ravens_numerical.eval.tasks.legacy.rules import RulesTask

    cfg = Config(models=["m"], backend="vllm", task_type="aba")
    backend = _make_backend(
        cfg, "EleutherAI/pythia-160m-deduped", RulesTask(), score_mode="forced_choice"
    )
    assert isinstance(backend, VLLMBackend)
    assert not isinstance(backend, ForcedChoiceVLLMBackend)


def test_make_backend_forced_choice_hierarchical_uses_guided_backend():
    from ravens_numerical.eval.tasks.legacy.hierarchical import HierarchicalTask

    cfg = Config(models=["m"], backend="vllm", task_type="hierarchical")
    backend = _make_backend(
        cfg,
        "EleutherAI/pythia-160m-deduped",
        HierarchicalTask(),
        score_mode="forced_choice",
    )
    assert isinstance(backend, ForcedChoiceVLLMBackend)


def test_make_backend_forced_choice_ravens_uses_plain_vllm():
    """Raven completion options like ``12]`` 500 under vLLM structured choice."""
    from ravens_numerical.eval.backends import VLLMBackend
    from ravens_numerical.eval.tasks.ravens import RavensNumericalTask

    cfg = Config(
        models=["m"],
        backend="vllm",
        task_type="ravens",
        ravens_prompt_type="completion",
    )
    task = RavensNumericalTask(prompt_type="completion")
    backend = _make_backend(
        cfg, "EleutherAI/pythia-1b-deduped", task, score_mode="forced_choice"
    )
    assert isinstance(backend, VLLMBackend)
    assert not isinstance(backend, ForcedChoiceVLLMBackend)


def test_make_backend_forced_choice_webb_instruction_uses_plain_vllm():
    from ravens_numerical.eval.backends import VLLMBackend
    from ravens_numerical.eval.cli import _instantiate_task

    cfg = Config(
        models=["m"],
        backend="vllm",
        task_type="webb",
        ravens_prompt_type="instruction",
        ravens_max_tasks=1,
        score_mode="forced_choice",
    )
    task = _instantiate_task("ravens_numerical", cfg, score_mode="forced_choice")
    backend = _make_backend(
        cfg, "BabyLM-community/babylm-baseline-10m-gpt2", task, score_mode="forced_choice"
    )
    assert isinstance(backend, VLLMBackend)
    assert not isinstance(backend, PythiaChoiceOnlyVLLMBackend)


def test_make_backend_webb_choice_only_same_as_ravens():
    from ravens_numerical.paths import TASKS_WEBB_JSON

    cfg = Config(
        models=["EleutherAI/pythia-410m-deduped"],
        tasks=["ravens_numerical"],
        task_type="webb",
        ravens_prompt_type="instruction",
        ravens_prompt_mode="choice_only",
        score_mode="free_gen",
    )
    task = RavensNumericalTask(
        tasks_json=TASKS_WEBB_JSON,
        max_tasks=1,
        prompt_type="instruction",
        prompt_mode="choice_only",
        load_icl_from_json=True,
    )
    assert isinstance(_make_backend(cfg, cfg.models[0], task), PythiaChoiceOnlyVLLMBackend)
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=1)
    assert '"choice"' in prompt
    assert "Options (each has a letter A, B, C, or D)" in prompt


def test_make_backend_miniberta_aba_uses_completion_not_choice_only():
    cfg = Config(
        models=["nyu-mll/roberta-base-100M-1"],
        backend="hf",
        task_type="aba",
        ravens_prompt_type="instruction",
        ravens_prompt_mode="auto",
    )
    from ravens_numerical.eval.tasks.legacy.rules import RulesTask

    backend = _make_backend(cfg, cfg.models[0], RulesTask())
    assert isinstance(backend, RobertaMLMBackend)
    assert backend._prompt_type == "completion"
    assert backend._prompt_mode == "plain"


def test_make_backend_miniberta_hierarchical_uses_completion():
    cfg = Config(
        models=["nyu-mll/roberta-base-100M-1"],
        backend="hf",
        task_type="hierarchical",
        ravens_prompt_type="instruction",
        ravens_prompt_mode="choice_only",
    )
    from ravens_numerical.eval.tasks.legacy.hierarchical import HierarchicalTask

    backend = _make_backend(cfg, cfg.models[0], HierarchicalTask())
    assert isinstance(backend, RobertaMLMBackend)
    assert backend._prompt_type == "completion"


def test_roberta_generate_aba_returns_syllable_not_choice_json(mocker):
    from ravens_numerical.eval.tasks.base import Stimulus
    from ravens_numerical.eval.tasks.legacy.rules import RulesTask

    backend = RobertaMLMBackend(
        "nyu-mll/roberta-base-100M-1",
        prompt_type="instruction",  # misconfigured; generate should still use choices
        prompt_mode="choice_only",
    )
    mocker.patch.object(
        backend,
        "score_completion",
        side_effect=lambda prompt, completion: {" pe": -1.0, " bi": -3.0}[completion],
    )
    stimulus = Stimulus(
        query="pe bi",
        expected="pe",
        answer_choices=["pe", "bi"],
        metadata={"rule": "ABA"},
    )
    result = backend.generate("pe bi", stimulus=stimulus, task=RulesTask())
    assert result.text == "pe"
    assert "choice" not in result.text


@resp.activate
def test_qwen3_choice_only_backend_uses_chat():
    backend = Qwen3ChoiceOnlyVLLMBackend("Qwen/Qwen3-8B", "http://localhost:8000")
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": '{"choice":"D"}'},
                    "logprobs": {"content": []},
                }
            ]
        },
        status=200,
    )
    result = backend.generate("prompt")
    assert result.text == '{"choice":"D"}'
    body = json.loads(resp.calls[0].request.body)
    assert body["messages"][0]["role"] == "user"
    assert body["response_format"]["type"] == "json_schema"


@resp.activate
def test_olmo2_base_choice_only_backend_uses_guided_choice():
    backend = Olmo2ChoiceOnlyVLLMBackend(
        "allenai/OLMo2-7B-1124", "http://localhost:8000"
    )
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/completions",
        json={
            "choices": [
                {
                    "text": "C",
                    "logprobs": {
                        "tokens": ["C"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [
                            {"A": -2.0, "B": -1.5, "C": -0.2, "D": -3.0},
                        ],
                    },
                }
            ]
        },
        status=200,
    )
    result = backend.generate(
        'Matrix:\n?\n\nYour entire reply must be a single JSON object with exactly one key "choice".'
    )
    assert result.text == '{"choice":"C"}'
    body = json.loads(resp.calls[0].request.body)
    assert "extra_body" not in body
    assert body["structured_outputs"]["choice"] == ["A", "B", "C", "D"]
    assert "single JSON object" not in body["prompt"]
    assert "Reply with ONLY the option letter" in body["prompt"]


@resp.activate
def test_olmo2_base_recovers_empty_json_from_logprobs():
    backend = Olmo2ChoiceOnlyVLLMBackend(
        "allenai/OLMo2-7B-1124", "http://localhost:8000"
    )
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/completions",
        json={
            "choices": [
                {
                    "text": '{"choice":""}',
                    "logprobs": {
                        "tokens": ['{"choice":""}'],
                        "token_logprobs": [-0.1],
                        "top_logprobs": [
                            {"A": -2.0, "B": -0.3, "C": -1.5, "D": -3.0},
                        ],
                    },
                }
            ]
        },
        status=200,
    )
    result = backend.generate("prompt")
    assert result.text == '{"choice":"B"}'


@resp.activate
def test_olmo2_instruct_choice_only_backend_uses_chat():
    backend = Olmo2InstructChoiceOnlyVLLMBackend(
        "allenai/OLMo-2-1124-7B-Instruct", "http://localhost:8000"
    )
    resp.add(
        resp.POST,
        "http://localhost:8000/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": '{"choice":"B"}'},
                    "logprobs": {"content": []},
                }
            ]
        },
        status=200,
    )
    result = backend.generate("prompt")
    assert result.text == '{"choice":"B"}'
    body = json.loads(resp.calls[0].request.body)
    assert body["messages"][0]["role"] == "user"
    assert body["response_format"]["type"] == "json_schema"


class _StubChoiceBackend(ModelBackend):
    def __init__(self) -> None:
        self._model = "stub"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        _ = kwargs
        raw = {
            "choices": [
                {
                    "text": '{"choice":"B"}',
                    "logprobs": {
                        "tokens": ['{"choice":"', "B", '"}'],
                        "token_logprobs": [-0.1, -0.2, -0.3],
                        "top_logprobs": [
                            {},
                            {"A": -2.0, "B": -0.2, "C": -3.0, "D": -4.0},
                            {},
                        ],
                    },
                }
            ]
        }
        return ModelResponse(text='{"choice":"B"}', raw=raw)

    def score_completion(self, prompt: str, completion: str):
        return None


def test_runner_records_choice_only_metrics():
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    results = evaluate(task, _StubChoiceBackend(), n_examples=0)
    assert len(results) == 1
    score = results[0].score
    assert score.correct is True
    assert score.logprob_argmax_correct is True
    assert score.brier is not None
    assert score.logprob_correct is None
