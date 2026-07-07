"""Tests for ravens_answer_parse."""

from ravens_numerical.parsing.answer_parse import parse_answer, parse_structured_choice


def test_parse_structured_choice_minimal():
    assert parse_structured_choice('{"choice":"B"}') == 1
    assert parse_structured_choice('{"choice": "c"}') == 2


def test_parse_structured_choice_fenced():
    text = '```json\n{"choice":"D"}\n```'
    assert parse_structured_choice(text) == 3


def test_parse_structured_choice_trailing_junk():
    assert parse_structured_choice('{"choice":"A"} extra') == 0


def test_parse_answer_prefers_json_over_last_letter():
    # Fallback would pick D from trailing text; JSON path should win.
    assert parse_answer('{"choice":"B"} therefore D') == 1


def test_parse_answer_fallback_last_letter():
    assert parse_answer("The answer is B") == 1


def test_parse_answer_value_match_fallback():
    opts = [4, 30, 1, 2]
    assert parse_answer("the cell contains 30", answer_options=opts, correct_index=1) == 1


def test_parse_answer_none_on_empty():
    assert parse_answer("") is None
    assert parse_answer("no letters here") is None


def test_parse_answer_boxed():
    assert parse_answer(r"\boxed{C}") == 2
