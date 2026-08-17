"""Tests for nested N-scaling pilots."""

from __future__ import annotations

import json
import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "babylm_finetune" / "scripts" / "write_n_scaling_pilots.py"


def _load_module():
    spec = spec_from_file_location("write_n_scaling_pilots", _SCRIPT)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp(row: dict) -> str:
    return json.dumps(
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
            "task_type": row["task_type"],
        },
        sort_keys=True,
    )


def test_nested_stratified_pilots_are_nested():
    mod = _load_module()
    rows = []
    types = [
        "combine",
        "constancy",
        "constancy_row",
        "intersection",
        "pattern",
        "pattern_tuple",
        "progression",
    ]
    for tt in types:
        for i in range(20):
            rows.append(
                {
                    "prompt": f"p-{tt}-{i}",
                    "completion": f"c-{tt}-{i}",
                    "task_type": tt,
                }
            )
    sizes = (14, 28, 56, 100)
    pilots = mod.nested_stratified_pilots(rows, sizes, random.Random(0))
    ordered = sorted(pilots)
    for a, b in zip(ordered, ordered[1:]):
        assert {_fp(r) for r in pilots[a]} <= {_fp(r) for r in pilots[b]}
        assert len(pilots[a]) == a
