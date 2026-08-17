"""Include / holdout flags accept the three challenge types."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODULE = _REPO / "babylm_finetune" / "training" / "task_filter.py"


def _load():
    spec = spec_from_file_location("task_filter", _MODULE)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_types_includes_challenge_subtasks():
    mod = _load()
    for tt in ("distribution_of_three", "progression_plus_n", "tuple_grid"):
        assert tt in mod.ALL_TYPES


def test_parse_types_accepts_challenge_names():
    mod = _load()
    assert mod.parse_types("tuple_grid,progression_plus_n") == [
        "tuple_grid",
        "progression_plus_n",
    ]


def test_holdout_challenge_type():
    mod = _load()
    include, holdout = mod.resolve_include_types(
        holdout_types=["distribution_of_three"]
    )
    assert "distribution_of_three" in holdout
    assert "distribution_of_three" not in include
    assert "tuple_grid" in include


def test_unknown_type_rejected():
    mod = _load()
    with pytest.raises(ValueError, match="Unknown"):
        mod.parse_types("not_a_real_type")
