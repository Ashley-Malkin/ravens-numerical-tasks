import json
import dataclasses
from pathlib import Path
import pytest
from ravens_numerical.eval.tasks.base import (
    ModelBackend,
    ModelResponse,
    Stimulus,
    Task,
    TrialResult,
)
from ravens_numerical.eval.runner import evaluate, save_results


# --- Stubs ---


class StubBackend(ModelBackend):
    def __init__(
        self,
        text: str = "ro",
        logprob: float | None = -1.5,
        logprobs_by_completion: dict[str, float] | None = None,
    ):
        self._text = text
        self._logprob = logprob
        self._logprobs_by_completion = logprobs_by_completion or {}

    @property
    def model(self) -> str:
        return "stub-model"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(text=self._text)

    def score_completion(self, prompt: str, completion: str) -> float | None:
        if completion in self._logprobs_by_completion:
            return self._logprobs_by_completion[completion]
        return self._logprob


class StubTask(Task):
    def canonical_stimuli(self) -> list[Stimulus]:
        return [
            Stimulus(query="de ro ___", expected="ro", metadata={"rule": "ABB"}),
            Stimulus(query="de ro ___", expected="de", metadata={"rule": "ABA"}),
        ]

    def generate_stimulus(self, n_examples: int = 3) -> Stimulus:
        return Stimulus(query="x y ___", expected="y")

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        return response.text.strip().lower() == stimulus.expected.strip().lower()

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        return f"prompt: {stimulus.query}"


# --- Tests ---


def test_evaluate_returns_one_result_per_stimulus():
    results = evaluate(StubTask(), StubBackend(), n_examples=0)
    assert len(results) == 2


def test_evaluate_correct_when_response_matches():
    results = evaluate(StubTask(), StubBackend(text="ro"), n_examples=0)
    assert results[0].score.correct is True


def test_evaluate_incorrect_when_response_mismatches():
    results = evaluate(StubTask(), StubBackend(text="ro"), n_examples=0)
    assert results[1].score.correct is False


def test_evaluate_populates_logprob_correct():
    results = evaluate(StubTask(), StubBackend(logprob=-2.5), n_examples=0)
    assert results[0].score.logprob_correct == pytest.approx(-2.5)


def test_evaluate_handles_none_logprob():
    results = evaluate(StubTask(), StubBackend(logprob=None), n_examples=0)
    assert results[0].score.logprob_correct is None


def test_evaluate_uses_custom_stimuli():
    custom = [Stimulus(query="x y ___", expected="x")]
    results = evaluate(
        StubTask(), StubBackend(text="x"), n_examples=0, stimuli=custom
    )
    assert len(results) == 1
    assert results[0].score.correct is True


def test_evaluate_result_fields_populated():
    results = evaluate(StubTask(), StubBackend(), n_examples=3)
    r = results[0]
    assert r.task == "stub"
    assert r.n_examples == 3
    assert r.timestamp


def test_save_results_writes_json(tmp_path):
    results = evaluate(StubTask(), StubBackend(), n_examples=0)
    path = save_results(
        results,
        model="test_model",
        task="stub",
        n_examples=0,
        results_dir=tmp_path,
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2


def test_evaluate_free_gen_correct_when_generation_matches():
    # Default score_mode=free_gen: correctness from task.score() on generated text
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(text="1", logprobs_by_completion={"0": -3.0, "1": -1.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.correct is True


def test_evaluate_free_gen_correct_even_when_logprobs_disagree():
    # generate() returns "1" which matches expected; logprobs favor "0" but
    # free_gen correctness is based on generated text, not logprob argmax.
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(text="1", logprobs_by_completion={"0": -1.0, "1": -3.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.correct is True


def test_evaluate_free_gen_incorrect_when_generation_is_gibberish():
    # Logprob comparison favors expected "1", but generated text is gibberish
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(text="parem dem chamama", logprobs_by_completion={"0": -3.0, "1": -1.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.correct is False


def test_evaluate_forced_choice_uses_logprob_argmax():
    # When generation is not a valid choice, fall back to logprob argmax
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(
        text="parem dem chamama",
        logprobs_by_completion={"0": -3.0, "1": -1.0},
    )
    results = evaluate(
        StubTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is True


def test_evaluate_forced_choice_uses_echo_over_mismatched_generation():
    # Constrained gen says "0" but logprobs favor "1" → grade from echo argmax
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(
        text="0",
        logprobs_by_completion={"0": -3.0, "1": -1.0},
    )
    results = evaluate(
        StubTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is True


class ClosingBracketTask(StubTask):
    """Stub like Raven/matrix_easy completion (choice + ``]``)."""

    def format_completion(self, stimulus, choice):
        return choice + "]"


def test_evaluate_forced_choice_matches_formatted_completion_text():
    stimulus = Stimulus(
        query="[", expected="12", answer_choices=["12", "10", "8", "18"]
    )
    backend = StubBackend(
        text="12]",
        logprobs_by_completion={"12]": -1.0, "10]": -3.0, "8]": -4.0, "18]": -5.0},
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is True


def test_evaluate_forced_choice_does_not_hijack_long_cell_with_short_prefix():
    """Multi-token cell ``5 7 4 1]...`` must not match distractor ``5`` via first token."""
    stimulus = Stimulus(
        query="[",
        expected="5 7 4 1",
        answer_choices=["5", "7 4", "5 4 1", "5 7 4 1"],
    )
    backend = StubBackend(
        text="5 7 4 1]\n\n[6] [7] [6 7]\n[",
        logprobs_by_completion={
            "5]": -8.0,
            "7 4]": -9.0,
            "5 4 1]": -7.0,
            "5 7 4 1]": -4.0,
        },
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is True


def test_evaluate_forced_choice_echo_beats_short_generation_match():
    """Scalar gen match must not override a length-normalized pair logprob win."""
    stimulus = Stimulus(
        query="[",
        expected="10 11",
        answer_choices=["10", "7 10", "11 10", "10 11"],
    )
    backend = StubBackend(
        text="10]",
        logprobs_by_completion={
            "10]": -0.10,
            "7 10]": -3.0,
            "11 10]": -3.0,
            "10 11]": -0.12,
        },
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.logprob_argmax_correct is True
    assert results[0].score.correct is True
    """If span is multi-token gibberish starting with a short choice, use logprobs."""
    stimulus = Stimulus(
        query="[",
        expected="5 7 4 1",
        answer_choices=["5", "7 4", "5 4 1", "5 7 4 1"],
    )
    backend = StubBackend(
        text="5 9 9]\n[",
        logprobs_by_completion={
            "5]": -8.0,
            "7 4]": -9.0,
            "5 4 1]": -7.0,
            "5 7 4 1]": -4.0,
        },
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    # No exact choice match → echo logprob argmax (correct option).
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is True


def test_evaluate_forced_choice_incorrect_when_logprobs_favor_wrong():
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(
        text="gibberish",
        logprobs_by_completion={"0": -1.0, "1": -3.0},
    )
    results = evaluate(
        StubTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is False
    assert results[0].score.logprob_argmax_correct is False


def test_evaluate_forced_choice_abstains_on_tied_echo_logprobs():
    """All-equal echo scores (e.g. empty-sum artifact as 0.0) must not pick first key."""
    stimulus = Stimulus(
        query="[", expected="185", answer_choices=["183", "186", "185", "187"]
    )
    backend = StubBackend(
        text="15151]",
        logprobs_by_completion={
            "183]": 0.0,
            "186]": 0.0,
            "185]": 0.0,
            "187]": 0.0,
        },
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is False
    assert results[0].score.logprob_argmax_correct is None


def test_evaluate_forced_choice_tied_echo_still_uses_generation_match():
    stimulus = Stimulus(
        query="[", expected="185", answer_choices=["183", "186", "185", "187"]
    )
    backend = StubBackend(
        text="185]",
        logprobs_by_completion={
            "183]": 0.0,
            "186]": 0.0,
            "185]": 0.0,
            "187]": 0.0,
        },
    )
    results = evaluate(
        ClosingBracketTask(),
        backend,
        n_examples=0,
        stimuli=[stimulus],
        score_mode="forced_choice",
    )
    assert results[0].score.correct is True
    assert results[0].score.logprob_argmax_correct is None


def test_evaluate_forced_choice_stores_logprob_of_expected():
    stimulus = Stimulus(
        query="AABB ", expected="1", answer_choices=["0", "1"]
    )
    backend = StubBackend(logprobs_by_completion={"0": -3.0, "1": -1.5})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.logprob_correct == pytest.approx(-1.5)


def test_evaluate_without_answer_choices_uses_free_generation():
    # Existing behavior preserved when answer_choices is None
    stimulus = Stimulus(query="de ro", expected="ro")
    results = evaluate(StubTask(), StubBackend(text="ro"), n_examples=0, stimuli=[stimulus])
    assert results[0].score.correct is True


class SpacePrefixTask(StubTask):
    """StubTask that prepends a space to completions (like rules/hierarchical)."""

    def format_completion(self, stimulus, choice):
        return " " + choice


def test_evaluate_forced_choice_uses_format_completion_for_logprobs():
    """score_completion receives the formatted choice (with space prefix)."""
    stimulus = Stimulus(
        query="AABB", expected="1", answer_choices=["0", "1"]
    )
    # Backend keyed by formatted choices; default logprob is -99
    backend = StubBackend(
        text="1",
        logprob=-99.0,
        logprobs_by_completion={" 0": -3.0, " 1": -1.0},
    )
    results = evaluate(
        SpacePrefixTask(), backend, n_examples=0, stimuli=[stimulus]
    )
    # logprob_correct should use formatted key " 1", not bare "1"
    assert results[0].score.logprob_correct == pytest.approx(-1.0)


def test_evaluate_free_gen_uses_format_completion_for_logprob():
    stimulus = Stimulus(query="de ro", expected="ro")
    backend = StubBackend(text="ro", logprobs_by_completion={" ro": -2.0}, logprob=-99.0)
    results = evaluate(
        SpacePrefixTask(), backend, n_examples=0, stimuli=[stimulus]
    )
    assert results[0].score.logprob_correct == pytest.approx(-2.0)


def test_evaluate_prob_correct_is_softmax_of_logprobs():
    import math
    stimulus = Stimulus(query="AABB ", expected="1", answer_choices=["0", "1"])
    backend = StubBackend(logprobs_by_completion={"0": -3.0, "1": -1.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    # softmax: exp(-1) / (exp(-3) + exp(-1))
    expected_prob = math.exp(-1.0) / (math.exp(-3.0) + math.exp(-1.0))
    assert results[0].score.prob_correct == pytest.approx(expected_prob)


def test_evaluate_prob_correct_lower_when_not_best_choice():
    import math
    stimulus = Stimulus(query="AABB ", expected="1", answer_choices=["0", "1"])
    backend = StubBackend(logprobs_by_completion={"0": -1.0, "1": -3.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    expected_prob = math.exp(-3.0) / (math.exp(-1.0) + math.exp(-3.0))
    assert results[0].score.prob_correct == pytest.approx(expected_prob)


def test_evaluate_prob_correct_is_none_without_answer_choices():
    stimulus = Stimulus(query="de ro", expected="ro")
    results = evaluate(StubTask(), StubBackend(text="ro"), n_examples=0, stimuli=[stimulus])
    assert results[0].score.prob_correct is None


def test_evaluate_answer_logprobs_stores_all_choices():
    stimulus = Stimulus(query="AABB ", expected="1", answer_choices=["0", "1"])
    backend = StubBackend(logprobs_by_completion={"0": -3.0, "1": -1.0})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.answer_logprobs == {"0": -3.0, "1": -1.0}


def test_evaluate_prob_correct_is_none_when_all_logprobs_are_none():
    stimulus = Stimulus(query="AABB ", expected="1", answer_choices=["0", "1"])
    backend = StubBackend(logprob=None, logprobs_by_completion={})
    results = evaluate(StubTask(), backend, n_examples=0, stimuli=[stimulus])
    assert results[0].score.prob_correct is None


def test_evaluate_answer_logprobs_is_none_without_answer_choices():
    stimulus = Stimulus(query="de ro", expected="ro")
    results = evaluate(StubTask(), StubBackend(text="ro"), n_examples=0, stimuli=[stimulus])
    assert results[0].score.answer_logprobs is None


def test_save_results_path_structure(tmp_path):
    run_id = "2026-03-25T17-35-08"
    results = evaluate(StubTask(), StubBackend(), n_examples=0)
    path = save_results(
        results,
        model="llama3:8b",
        task="rules",
        n_examples=0,
        results_dir=tmp_path,
        run_id=run_id,
    )
    # Structure: <results_dir>/<model>/<run_id>/<task>/<n_examples>_examples.json
    assert "llama3_8b" in str(path)
    assert run_id in str(path)
    assert path.name == "0_examples.json"
    assert path.parent.name == "rules"
    assert path.parent.parent.name == run_id


def test_save_results_run_id_groups_results(tmp_path):
    run_id = "2026-03-25T12-00-00"
    results = evaluate(StubTask(), StubBackend(), n_examples=0)
    p1 = save_results(results, model="m", task="rules", n_examples=0,
                       results_dir=tmp_path, run_id=run_id)
    p2 = save_results(results, model="m", task="rules", n_examples=3,
                       results_dir=tmp_path, run_id=run_id)
    # Both files share the same run directory
    assert p1.parent == p2.parent
    assert p1.name == "0_examples.json"
    assert p2.name == "3_examples.json"
