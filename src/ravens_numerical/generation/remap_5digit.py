"""Build a 5-digit OOD copy of ``data/tasks.json`` for generalization evals.

Every integer leaf ``n`` is remapped to ``10000 + n`` (so values stay in
``[10000, 10499]``, all five digits). Matrices, equality structure, and
``correct_index`` / ``correct_letter`` are unchanged — only the surface
magnitude shifts. Tuple cells keep the same list shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ravens_numerical.paths import REPO_ROOT, TASKS_JSON

FIVE_DIGIT_OFFSET = 10_000
DEFAULT_OUT = REPO_ROOT / "data" / "tasks_5digit.json"


def remap_value(value: Any, *, offset: int = FIVE_DIGIT_OFFSET) -> Any:
    """Recursively add ``offset`` to every integer leaf (bools untouched)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return offset + value
    if isinstance(value, list):
        return [remap_value(v, offset=offset) for v in value]
    if isinstance(value, dict):
        return {k: remap_value(v, offset=offset) for k, v in value.items()}
    return value


def remap_task(task: dict[str, Any], *, offset: int = FIVE_DIGIT_OFFSET) -> dict[str, Any]:
    """Remap numeric leaves in matrix / answer_options; keep metadata fields."""
    out = dict(task)
    if "matrix" in out:
        out["matrix"] = remap_value(out["matrix"], offset=offset)
    if "answer_options" in out:
        out["answer_options"] = remap_value(out["answer_options"], offset=offset)
    return out


def build_tasks_5digit(
    source: Path | None = None,
    *,
    offset: int = FIVE_DIGIT_OFFSET,
) -> list[dict[str, Any]]:
    """Load ``tasks.json`` and return the 5-digit remapped task list."""
    path = source or TASKS_JSON
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = raw["tasks"] if isinstance(raw, dict) else raw
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a task list in {path}")
    return [remap_task(t, offset=offset) for t in tasks]


def write_tasks_5digit(
    output: Path | None = None,
    source: Path | None = None,
    *,
    offset: int = FIVE_DIGIT_OFFSET,
) -> Path:
    """Write ``tasks_5digit.json``; return the output path."""
    out = output or DEFAULT_OUT
    tasks = build_tasks_5digit(source, offset=offset)
    payload = {
        "tasks": tasks,
        "meta": {
            "source": str((source or TASKS_JSON).as_posix()),
            "transform": f"integer_leaf -> {offset} + leaf",
            "purpose": "OOD magnitude generalization (5-digit surface form)",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap data/tasks.json integers to 5-digit form for OOD eval."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=TASKS_JSON,
        help=f"Source tasks JSON (default: {TASKS_JSON})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=FIVE_DIGIT_OFFSET,
        help=f"Added to every integer leaf (default: {FIVE_DIGIT_OFFSET})",
    )
    args = parser.parse_args()
    path = write_tasks_5digit(args.output, args.source, offset=args.offset)
    n = len(json.loads(path.read_text(encoding="utf-8"))["tasks"])
    print(f"Wrote {n} tasks → {path}")


if __name__ == "__main__":
    main()
