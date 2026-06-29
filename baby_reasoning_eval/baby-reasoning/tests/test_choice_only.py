"""Tests for instruction choice-only wiring."""

import json

import pytest
import responses as resp

from baby_reasoning.cli import Config, _make_backend, _results_suffix
from baby_reasoning.model import (
    PythiaChoiceOnlyVLLMBackend,
    Qwen3ChoiceOnlyVLLMBackend,
    strip_qwen_thinking,
)
from baby_reasoning.runner import evaluate
from baby_reasoning.tasks.base import ModelBackend, ModelResponse
from baby_reasoning.tasks.ravens_numerical import RavensNumericalTask


def test_ravens_task_uses_choice_only_prompt_mode():
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    assert task.uses_choice_only_metrics is True
    stimulus = task.canonical_stimuli()[0]
    prompt = task.build_prompt(stimulus, n_examples=0)
    assert "single JSON object" in prompt
    assert '"choice"' in prompt


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
        ravens_prompt_mode="auto",
    )
    assert _results_suffix(cfg, "ravens_numerical") == "choice_only"


def test_make_backend_selects_family():
    cfg = Config(
        models=["EleutherAI/pythia-410m-deduped"],
        tasks=["ravens_numerical"],
        ravens_prompt_mode="choice_only",
    )
    task = RavensNumericalTask(max_tasks=1, prompt_mode="choice_only")
    assert isinstance(_make_backend(cfg, cfg.models[0], task), PythiaChoiceOnlyVLLMBackend)

    cfg.models = ["Qwen/Qwen3-8B"]
    assert isinstance(_make_backend(cfg, cfg.models[0], task), Qwen3ChoiceOnlyVLLMBackend)


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
    assert body["extra_body"]["guided_json"]["required"] == ["choice"]


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


class _StubChoiceBackend(ModelBackend):
    def __init__(self) -> None:
        self._model = "stub"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> ModelResponse:
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
