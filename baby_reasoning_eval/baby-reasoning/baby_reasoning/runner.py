from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from baby_reasoning import RESULTS_DIR
from baby_reasoning.tasks.base import (
    ModelBackend,
    Stimulus,
    Task,
    TrialResult,
    TrialScore,
)


def _task_name(task: Task) -> str:
    """Derive snake-case task name from class name, e.g. RulesTask → 'rules'."""
    name = type(task).__name__
    name = name.removesuffix("Task")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _repo_root_from_task(task: Task) -> Path | None:
    repo_root = getattr(task, "_repo_root", None)
    if repo_root is not None:
        return Path(repo_root)
    return None


def _choice_only_metrics(
    task: Task,
    response,
    stimulus: Stimulus,
) -> tuple[bool | None, float | None]:
    if not getattr(task, "uses_choice_only_metrics", False) or response.raw is None:
        return None, None

    repo_root = _repo_root_from_task(task)
    if repo_root is not None:
        root_s = str(repo_root.resolve())
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    from ravens_choice_logprobs import (
        has_finite_letter_logprob,
        letter_logprobs_to_probs,
        multiclass_brier_score,
        parse_logprobs_by_letter_vllm,
    )

    pred_argmax, logprobs_by_letter = parse_logprobs_by_letter_vllm(
        response.raw, response.text
    )
    task_dict = stimulus.metadata.get("task", {})
    correct_idx = int(task_dict.get("correct_index", 0))
    logprob_argmax_correct = (
        pred_argmax == correct_idx if pred_argmax is not None else False
    )
    brier = None
    if has_finite_letter_logprob(logprobs_by_letter):
        probs = letter_logprobs_to_probs(logprobs_by_letter)
        if probs is not None and 0 <= correct_idx < len(probs):
            brier = multiclass_brier_score(probs, correct_idx)
    return logprob_argmax_correct, brier


def evaluate(
    task: Task,
    backend: ModelBackend,
    n_examples: int,
    stimuli: list[Stimulus] | None = None,
) -> list[TrialResult]:
    if stimuli is None:
        stimuli = task.canonical_stimuli()

    task_name = _task_name(task)
    results = []
    choice_only = getattr(task, "uses_choice_only_metrics", False)

    for stimulus in stimuli:
        prompt = task.build_prompt(stimulus, n_examples)
        response = backend.generate(prompt)
        correct = task.score(response, stimulus)
        logprob_argmax_correct, brier = _choice_only_metrics(task, response, stimulus)

        if choice_only and response.raw is not None:
            logprob_correct = None
            prob_correct = None
            answer_logprobs = None
        elif stimulus.answer_choices is not None:
            logprobs = {
                c: backend.score_completion(prompt, task.format_completion(stimulus, c))
                for c in stimulus.answer_choices
            }
            logprob_correct = logprobs.get(stimulus.expected)
            valid = {c: lp for c, lp in logprobs.items() if lp is not None}
            if valid:
                max_lp = max(valid.values())
                denom = sum(math.exp(lp - max_lp) for lp in valid.values())
                lp_c = valid.get(stimulus.expected)
                prob_correct = math.exp(lp_c - max_lp) / denom if lp_c is not None else None
            else:
                prob_correct = None
            answer_logprobs = logprobs
        else:
            logprob_correct = backend.score_completion(
                prompt, task.format_completion(stimulus, stimulus.expected)
            )
            prob_correct = None
            answer_logprobs = None

        score = TrialScore(
            correct=correct,
            logprob_correct=logprob_correct,
            prob_correct=prob_correct,
            answer_logprobs=answer_logprobs,
            logprob_argmax_correct=logprob_argmax_correct,
            brier=brier,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        results.append(
            TrialResult(
                model=backend.model,
                task=task_name,
                n_examples=n_examples,
                stimulus=stimulus,
                response=response,
                score=score,
                timestamp=timestamp,
            )
        )

    return results


def save_results(
    results: list[TrialResult],
    model: str,
    task: str,
    n_examples: int,
    results_dir: Path | None = None,
    run_id: str | None = None,
    results_suffix: str | None = None,
) -> Path:
    if results_dir is None:
        results_dir = RESULTS_DIR

    model_tag = model.replace(":", "_").replace("/", "--")
    if run_id is None:
        run_id = (
            results[0].timestamp.replace(":", "-")
            if results
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        )

    out_dir = results_dir / model_tag / run_id / task
    out_dir.mkdir(parents=True, exist_ok=True)
    if results_suffix:
        out_path = out_dir / f"{n_examples}_examples_{results_suffix}.json"
    else:
        out_path = out_dir / f"{n_examples}_examples.json"

    data = [dataclasses.asdict(r) for r in results]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
