from pathlib import Path

import pytest

from ravens_numerical.eval.cli import (
    Config,
    TASK_MAP,
    resolve_n_examples,
    resolve_ravens_prompt_type,
    resolve_score_mode,
    resolve_tasks,
    run,
    _instantiate_task,
    _results_suffix,
)
from ravens_numerical.eval.tasks.legacy.hierarchical import HierarchicalTask
from ravens_numerical.eval.tasks.legacy.rules import RulesTask
from ravens_numerical.paths import TASKS_ABA_JSON


# ---------------------------------------------------------------------------
# TASK_MAP
# ---------------------------------------------------------------------------

def test_task_map_has_all_tasks():
    assert set(TASK_MAP) == {"rules", "hierarchical", "matrix", "matrix_easy"}


def test_task_map_instantiates():
    for name, cls in TASK_MAP.items():
        task = cls()
        assert hasattr(task, "canonical_stimuli")


# ---------------------------------------------------------------------------
# resolve_tasks / task_type / n_examples
# ---------------------------------------------------------------------------

def test_resolve_tasks_ravens_default():
    cfg = Config(models=["m1"], task_type="ravens")
    assert resolve_tasks(cfg) == ["ravens_numerical"]


def test_resolve_tasks_aba():
    cfg = Config(models=["m1"], task_type="aba")
    assert resolve_tasks(cfg) == ["rules"]


def test_resolve_tasks_hierarchical():
    cfg = Config(models=["m1"], task_type="hierarchical")
    assert resolve_tasks(cfg) == ["hierarchical"]


def test_resolve_tasks_webb():
    cfg = Config(models=["m1"], task_type="webb")
    assert resolve_tasks(cfg) == ["ravens_numerical"]


def test_resolve_n_examples_webb_default():
    cfg = Config(models=["m1"], task_type="webb")
    assert resolve_n_examples(cfg) == [0, 3]


def test_resolve_tasks_explicit_overrides_suite():
    cfg = Config(models=["m1"], task_type="ravens", tasks=["matrix_easy"])
    assert resolve_tasks(cfg) == ["matrix_easy"]


def test_resolve_n_examples_ravens_default():
    cfg = Config(models=["m1"], task_type="ravens")
    assert resolve_n_examples(cfg) == [0, 3]


def test_resolve_n_examples_aba_default():
    cfg = Config(models=["m1"], task_type="aba")
    assert resolve_n_examples(cfg) == [0, 5, 10, 15, 20]


def test_resolve_n_examples_hierarchical_default():
    cfg = Config(models=["m1"], task_type="hierarchical")
    assert resolve_n_examples(cfg) == [0, 5, 10, 15, 20]


def test_resolve_n_examples_aba_rejects_unsupported():
    cfg = Config(models=["m1"], task_type="aba", n_examples=[0, 3])
    with pytest.raises(ValueError, match="only supports"):
        resolve_n_examples(cfg)


def test_config_defaults_completion_and_forced_choice():
    cfg = Config(models=["m"])
    assert cfg.ravens_prompt_type == "completion"
    assert cfg.score_mode == "forced_choice"
    assert resolve_score_mode(cfg, "EleutherAI/pythia-70m-deduped") == "forced_choice"
    assert resolve_ravens_prompt_type(cfg, "forced_choice") == "completion"


def test_resolve_score_mode_auto_pythia_checkpoint():
    cfg = Config(models=["m"], score_mode="auto")
    assert (
        resolve_score_mode(cfg, "EleutherAI/pythia-160m-deduped@step10000")
        == "forced_choice"
    )


def test_resolve_score_mode_auto_full_pythia():
    cfg = Config(models=["m"], score_mode="auto")
    assert resolve_score_mode(cfg, "EleutherAI/pythia-160m-deduped") == "free_gen"


def test_resolve_score_mode_explicit():
    cfg = Config(models=["m"], score_mode="forced_choice")
    assert resolve_score_mode(cfg, "EleutherAI/pythia-160m-deduped") == "forced_choice"
    cfg = Config(models=["m"], score_mode="free_gen")
    assert (
        resolve_score_mode(cfg, "EleutherAI/pythia-160m-deduped@step10000")
        == "free_gen"
    )


def test_resolve_ravens_prompt_type_forced_choice_uses_completion():
    cfg = Config(
        models=["m"],
        task_type="webb",
        ravens_prompt_type="instruction",
        score_mode="forced_choice",
    )
    assert resolve_ravens_prompt_type(cfg, "forced_choice") == "completion"
    assert resolve_ravens_prompt_type(cfg, "free_gen") == "instruction"


def test_forced_choice_webb_instantiates_completion_task():
    cfg = Config(
        models=["m"],
        task_type="webb",
        ravens_prompt_type="instruction",
        ravens_max_tasks=1,
        score_mode="forced_choice",
    )
    task = _instantiate_task("ravens_numerical", cfg, score_mode="forced_choice")
    assert task._prompt_type == "completion"
    stimulus = task.canonical_stimuli()[0]
    assert stimulus.answer_choices is not None


def test_results_suffix_forced_choice_ravens_uses_completion():
    cfg = Config(
        models=["m"],
        task_type="webb",
        ravens_prompt_type="instruction",
        score_mode="forced_choice",
    )
    assert _results_suffix(cfg, "ravens_numerical", "forced_choice") == "completion"


def test_results_suffix_forced_choice_rules():
    cfg = Config(models=["m"], task_type="aba")
    assert _results_suffix(cfg, "rules", "forced_choice") == "forced_choice"
    assert _results_suffix(cfg, "rules", "free_gen") is None


def test_run_task_type_aba_evaluates_rules(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], task_type="aba", n_examples=[0])
    run(cfg)

    assert mock_evaluate.call_count == 1
    assert type(mock_evaluate.call_args[0][0]).__name__ == "RulesTask"


def test_run_task_type_hierarchical_evaluates_hierarchical(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.ForcedChoiceVLLMBackend")

    cfg = Config(models=["m1"], task_type="hierarchical", n_examples=[0])
    run(cfg)

    assert mock_evaluate.call_count == 1
    assert type(mock_evaluate.call_args[0][0]).__name__ == "HierarchicalTask"


def test_run_task_type_ravens_default(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["EleutherAI/pythia-70m-deduped"], task_type="ravens", n_examples=[0])
    run(cfg)

    assert mock_evaluate.call_count == 1
    task = mock_evaluate.call_args[0][0]
    assert type(task).__name__ == "RavensNumericalTask"
    assert task._prompt_type == "completion"
    assert mock_evaluate.call_args.kwargs.get("score_mode") == "forced_choice"


def test_run_task_type_webb_loads_webb_json(mocker):
    from ravens_numerical.paths import TASKS_WEBB_JSON

    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(
        models=["EleutherAI/pythia-70m-deduped"],
        task_type="webb",
        n_examples=[0],
        ravens_max_tasks=2,
    )
    run(cfg)

    task = mock_evaluate.call_args[0][0]
    assert type(task).__name__ == "RavensNumericalTask"
    assert task._tasks_path == TASKS_WEBB_JSON
    assert task._icl_examples is not None
    assert task._prompt_type == "completion"


# ---------------------------------------------------------------------------
# tasks_aba.json loaders
# ---------------------------------------------------------------------------

def test_rules_loads_from_tasks_aba_json():
    task = RulesTask(tasks_json=TASKS_ABA_JSON)
    stimuli = task.canonical_stimuli()
    assert len(stimuli) == 70
    assert {s.metadata["rule"] for s in stimuli} == {"ABA", "ABB"}
    for s in stimuli:
        assert len(s.few_shot_examples) == 20


def test_hierarchical_loads_from_tasks_aba_json():
    task = HierarchicalTask(tasks_json=TASKS_ABA_JSON)
    stimuli = task.canonical_stimuli()
    assert len(stimuli) == 70
    patterns = {s.metadata["pattern"] for s in stimuli}
    assert "same-same" in patterns
    assert "same-different" in patterns
    assert "different-different" in patterns
    for s in stimuli:
        assert len(s.few_shot_examples) == 20


def test_aba_icl_disjoint_from_eval():
    import json

    data = json.loads(TASKS_ABA_JSON.read_text())
    rules_q = {item["query"] for item in data["rules"]}
    for rule, exs in data["rules_icl"].items():
        icl_q = {q for q, _ in exs}
        assert rules_q.isdisjoint(icl_q), rule
        assert len(exs) == 20
    hier_q = {item["query"] for item in data["hierarchical"]}
    hier_icl_q = {q for q, _ in data["hierarchical_icl"]}
    assert hier_q.isdisjoint(hier_icl_q)
    assert len(data["hierarchical_icl"]) == 20


def test_aba_rules_icl_syllables_disjoint_from_eval():
    import json

    data = json.loads(TASKS_ABA_JSON.read_text())
    eval_syl = set()
    for item in data["rules"]:
        eval_syl.update(item["query"].split())
    icl_syl = set()
    for exs in data["rules_icl"].values():
        for q, _ in exs:
            icl_syl.update(q.split())
    assert eval_syl.isdisjoint(icl_syl)


# ---------------------------------------------------------------------------
# run() — combinations
# ---------------------------------------------------------------------------

def test_run_calls_evaluate_for_each_combination(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1", "m2"], tasks=["rules"], n_examples=[0])
    run(cfg)

    assert mock_evaluate.call_count == 2  # 2 models × 1 task × 1 n_examples


def test_run_calls_save_results_for_each_combination(mocker):
    mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mock_save = mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules", "hierarchical"], n_examples=[0, 5])
    run(cfg)

    assert mock_save.call_count == 4  # 1 model × 2 tasks × 2 n_examples


def test_run_passes_n_examples_to_evaluate(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[5])
    run(cfg)

    args = mock_evaluate.call_args[0]
    assert args[2] == 5  # n_examples is third positional arg


def test_run_uses_canonical_stimuli_by_default(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0])
    run(cfg)

    # stimuli arg should be None so evaluate uses canonical_stimuli internally
    assert mock_evaluate.call_args[0][3] is None


def test_run_generates_n_stimuli_when_specified(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0], n_stimuli=5)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 5


def test_run_reuses_stimuli_across_n_examples(mocker):
    """Same stimuli list is passed to evaluate for all n_examples values."""
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0, 5], n_stimuli=5)
    run(cfg)

    assert mock_evaluate.call_count == 2
    stimuli_0 = mock_evaluate.call_args_list[0][0][3]
    stimuli_5 = mock_evaluate.call_args_list[1][0][3]
    assert stimuli_0 is stimuli_5


def test_run_systematic_uses_systematic_stimuli(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0, 5], n_stimuli=10, systematic=True)
    run(cfg)

    # systematic_stimuli for rules returns n_per_rule × 2 rules = 20 stimuli
    stimuli_arg = mock_evaluate.call_args_list[0][0][3]
    assert stimuli_arg is not None
    assert len(stimuli_arg) == 20  # 10 per rule × 2 rules


def test_run_uses_all_stimuli_for_matrix(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["matrix"], n_examples=[0], n_stimuli=5, systematic=True)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    from ravens_numerical.eval.tasks.legacy.matrix import MatrixTask
    assert stimuli_arg is not None
    assert len(stimuli_arg) == len(MatrixTask().all_stimuli())


def test_run_uses_all_stimuli_for_matrix_easy(mocker):
    mock_evaluate = mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    cfg = Config(models=["m1"], tasks=["matrix_easy"], n_examples=[0], n_stimuli=5)
    run(cfg)

    stimuli_arg = mock_evaluate.call_args[0][3]
    from ravens_numerical.eval.tasks.legacy.matrix_easy import MatrixEasyTask
    assert stimuli_arg is not None
    assert len(stimuli_arg) == len(MatrixEasyTask().all_stimuli())


def test_run_passes_results_dir_to_save(mocker):
    mocker.patch("ravens_numerical.eval.cli.evaluate", return_value=[])
    mock_save = mocker.patch("ravens_numerical.eval.cli.save_results", return_value=Path("x"))
    mocker.patch("ravens_numerical.eval.cli.VLLMBackend")

    custom_dir = Path("/tmp/custom")
    cfg = Config(models=["m1"], tasks=["rules"], n_examples=[0], results_dir=custom_dir)
    run(cfg)

    assert mock_save.call_args[1]["results_dir"] == custom_dir
