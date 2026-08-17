from ravens_numerical.analysis.dump_scores import (
    _tag_matches_query,
    log_preamble_suite,
    overlay_regular_section,
    trial_correct,
)
from ravens_numerical.scoring.echo_logprobs import unique_logprob_argmax


def test_tag_matches_run_id_not_longer_budget():
    tag_10 = "--checkpoints--td-seed42-10M__all_types__20260806T211545Z"
    tag_100 = "--checkpoints--td-seed42-100M__all_types__20260806T212729Z"
    assert _tag_matches_query(tag_10, "td-seed42-10M__all_types__20260806T211545Z")
    assert _tag_matches_query(tag_10, "td-seed42-10M")
    assert not _tag_matches_query(tag_100, "td-seed42-10M")
    assert _tag_matches_query(tag_100, "td-seed42-100M")


def test_tag_matches_childes_volume_path():
    tag = (
        "--__modal--volumes--vo-x--hub--models--mcxfrank--childes-gpt2-ladder"
        "--snapshots--abc--development--seed42--rung1M"
    )
    assert _tag_matches_query(
        tag, "mcxfrank/childes-gpt2-ladder/development/seed42/rung1M"
    )
    assert not _tag_matches_query(
        tag, "mcxfrank/childes-gpt2-ladder/development/seed42/rung12M"
    )


def test_tag_matches_tinystories_base_folder():
    tag = "--ts-base--GPT2-small_tinystories_10m_1e-04"
    assert _tag_matches_query(tag, "tinystories/GPT2-small_tinystories_10m_1e-04")
    assert _tag_matches_query(tag, "GPT2-small_tinystories_10m_1e-04")
    tag = "BabyLM-community--babylm-baseline-100m-gpt2"
    assert _tag_matches_query(tag, "BabyLM-community/babylm-baseline-100m-gpt2")
    assert _tag_matches_query(
        "nyu-mll--roberta-base-100M-1", "nyu-mll/roberta-base-100M-1"
    )


def test_trial_correct_echo_beats_generation():
    row = {
        "stimulus": {
            "expected": "1 2",
            "answer_choices": ["1", "1 2", "2 1", "9 1"],
            "metadata": {"task": {"task_type": "tuple_grid"}},
        },
        "response": {"text": "1]"},
        "score": {
            "answer_logprobs": {
                "1": -0.9,
                "1 2": -1.58,
                "2 1": -10.0,
                "9 1": -18.0,
            }
        },
    }
    assert unique_logprob_argmax(row["score"]["answer_logprobs"]) == "1 2"
    assert trial_correct(row) is True


def test_log_preamble_suite_from_logged_line():
    text = (
        "_Logged x; eval data `tasks_5digit.json` (`--n-examples 0`)\n\n"
        "### mcxfrank/childes-gpt2-ladder/development/seed42/rung1M\n"
        "- **Overall:** 22.9%\n"
    )
    start = text.index("###")
    assert log_preamble_suite(text, start) == "5digit"
    scored = overlay_regular_section(text, start, "rung1M", 22.9, {"combine": 10.0})
    assert scored is None


def test_log_preamble_suite_combined_generated_header():
    text = (
        "_Generated 2026-08-12. eval data `challenge_tasks.json` (`--n-examples 0`)\n\n"
        "## `td-seed42-10M__all_types__20260806T211545Z`\n"
        "- **Overall:** 24.7%\n"
        "- **By task:** tuple_grid 52.0%\n"
    )
    start = text.index("## `")
    assert log_preamble_suite(text, start) == "challenge"
    scored = overlay_regular_section(
        text, start, "td-seed42-10M", 24.7, {"tuple_grid": 52.0}
    )
    assert scored is None


def test_trial_correct_collapsed_echo_uses_generation():
    row = {
        "stimulus": {
            "expected": "300",
            "answer_choices": ["300", "301", "299", "302"],
            "metadata": {"task": {"task_type": "constancy"}},
        },
        "response": {"text": "300]"},
        "score": {
            "answer_logprobs": {
                "300": -1e-6,
                "301": -2e-6,
                "299": -3e-6,
                "302": -4e-7,
            }
        },
    }
    assert trial_correct(row) is True
