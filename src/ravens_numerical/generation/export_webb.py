"""Export Webb et al. (2023) digit matrices to Raven-compatible JSON.

Canonical subset matches ``MatrixTask.canonical_stimuli``: up to five non-empty
test items per rule type (held-out last three reserved for ICL), excluding
``AND_permuted`` (empty answers). Each item's eight choices are reduced to four
deterministically (correct + three distractors, shuffled with a keyed RNG).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ravens_numerical.paths import LEGACY_DATA_DIR, TASKS_WEBB_JSON

_DEFAULT_NPZ = LEGACY_DATA_DIR / "matrix" / "all_problems.npz"
_DEFAULT_N_EXAMPLES = 3
_N_CANONICAL_PER_TYPE = 5
_N_OPTIONS = 4
_LETTERS = "ABCD"


def _cell_to_json(cell: Any) -> int | list[int] | None:
    """Convert an NPZ cell to JSON: scalar, list, or null (empty / all -1)."""
    arr = np.asarray(cell)
    values = [int(v) for v in arr.flat if int(v) != -1]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _option_key(option: Any) -> tuple:
    """Hashable key for equality / dedup of options."""
    val = _cell_to_json(option)
    if val is None:
        return ("empty",)
    if isinstance(val, list):
        return ("list", tuple(val))
    return ("int", val)


def _answer_is_empty(choice: Any) -> bool:
    return _cell_to_json(choice) is None


def _matrix_to_json(prob: Any) -> list[list[Any]]:
    """Convert a 3×3 NPZ problem to a JSON matrix; bottom-right is always null."""
    rows: list[list[Any]] = []
    for r in range(3):
        row: list[Any] = []
        for c in range(3):
            if r == 2 and c == 2:
                row.append(None)
            else:
                row.append(_cell_to_json(prob[r][c]))
        rows.append(row)
    return rows


def reduce_to_four_options(
    answer_choices: Any,
    correct_ind: int,
    *,
    rng: random.Random,
) -> tuple[list[Any], int, str]:
    """Select correct + 3 distractors and shuffle; return options, index, letter."""
    n = len(answer_choices)
    if not (0 <= correct_ind < n):
        raise ValueError(f"correct_ind {correct_ind} out of range for {n} choices")

    correct = answer_choices[correct_ind]
    if _answer_is_empty(correct):
        raise ValueError("cannot reduce empty correct answer")

    correct_key = _option_key(correct)
    distractors: list[Any] = []
    seen = {correct_key}
    for i, choice in enumerate(answer_choices):
        if i == correct_ind:
            continue
        if _answer_is_empty(choice):
            continue
        key = _option_key(choice)
        if key in seen:
            continue
        seen.add(key)
        distractors.append(choice)

    if len(distractors) < _N_OPTIONS - 1:
        raise ValueError(
            f"need at least {_N_OPTIONS - 1} unique non-empty distractors; "
            f"got {len(distractors)}"
        )

    selected = [_cell_to_json(correct)] + [
        _cell_to_json(d) for d in distractors[: _N_OPTIONS - 1]
    ]
    order = list(range(_N_OPTIONS))
    rng.shuffle(order)
    shuffled = [selected[i] for i in order]
    new_correct = order.index(0)
    return shuffled, new_correct, _LETTERS[new_correct]


def _seed_for(rule_type: str, index: int, purpose: str) -> int:
    """Stable 32-bit seed independent of PYTHONHASHSEED."""
    material = f"webb|{purpose}|{rule_type}|{index}".encode("utf-8")
    # FNV-1a 32-bit
    h = 2166136261
    for b in material:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _item_to_task(
    *,
    rule_type: str,
    index: int,
    prob: Any,
    answer_choices: Any,
    correct_ind: int,
    perm_invariant: bool,
    purpose: str,
) -> dict[str, Any]:
    rng = random.Random(_seed_for(rule_type, index, purpose))
    options, ci, letter = reduce_to_four_options(
        answer_choices, int(correct_ind), rng=rng
    )
    return {
        "task_type": rule_type,
        "matrix": _matrix_to_json(prob),
        "answer_options": options,
        "correct_index": ci,
        "correct_letter": letter,
        "perm_invariant": bool(perm_invariant),
        "webb_index": int(index),
        "webb_correct_ind": int(correct_ind),
    }


def export_webb_corpus(
    npz_path: Path | None = None,
    *,
    n_canonical_per_type: int = _N_CANONICAL_PER_TYPE,
    n_examples: int = _DEFAULT_N_EXAMPLES,
) -> dict[str, Any]:
    """Build ``{"tasks": [...], "icl": {...}}`` from the Webb NPZ."""
    path = Path(npz_path) if npz_path is not None else _DEFAULT_NPZ
    if not path.is_file():
        raise FileNotFoundError(f"Webb NPZ not found: {path}")

    all_problems = np.load(path, allow_pickle=True)["all_problems"].item()
    tasks: list[dict[str, Any]] = []
    icl: dict[str, list[dict[str, Any]]] = {}

    for rule_type, data in all_problems.items():
        probs = data["prob"]
        choices = data["answer_choices"]
        correct_inds = data["correct_ind"]
        perm_invariant = bool(data["perm_invariant"])
        n_total = len(probs)
        if n_total <= n_examples:
            continue

        # Held-out ICL pool: last n_examples
        icl_items: list[dict[str, Any]] = []
        for i in range(n_total - n_examples, n_total):
            correct = choices[i][int(correct_inds[i])]
            if _answer_is_empty(correct):
                continue
            try:
                icl_items.append(
                    _item_to_task(
                        rule_type=rule_type,
                        index=i,
                        prob=probs[i],
                        answer_choices=choices[i],
                        correct_ind=int(correct_inds[i]),
                        perm_invariant=perm_invariant,
                        purpose="icl",
                    )
                )
            except ValueError:
                continue
        if len(icl_items) < n_examples:
            # Skip types without a full usable ICL pool (e.g. AND_permuted).
            continue
        icl[rule_type] = icl_items

        # Canonical test items: first n_canonical_per_type non-empty from test range
        added = 0
        n_test = n_total - n_examples
        for i in range(n_test):
            if added >= n_canonical_per_type:
                break
            correct = choices[i][int(correct_inds[i])]
            if _answer_is_empty(correct):
                continue
            try:
                tasks.append(
                    _item_to_task(
                        rule_type=rule_type,
                        index=i,
                        prob=probs[i],
                        answer_choices=choices[i],
                        correct_ind=int(correct_inds[i]),
                        perm_invariant=perm_invariant,
                        purpose="task",
                    )
                )
            except ValueError:
                continue
            added += 1

    return {
        "source": "Webb et al. (2023) digit matrices",
        "npz": str(path.name),
        "n_options": _N_OPTIONS,
        "n_canonical_per_type": n_canonical_per_type,
        "n_icl_per_type": n_examples,
        "tasks": tasks,
        "icl": icl,
    }


def write_tasks_webb_json(
    output: Path | None = None,
    npz_path: Path | None = None,
) -> Path:
    """Export and write ``tasks_webb.json``; return the output path."""
    out = Path(output) if output is not None else TASKS_WEBB_JSON
    payload = export_webb_corpus(npz_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Webb digit matrices to data/tasks_webb.json"
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=_DEFAULT_NPZ,
        help=f"Path to all_problems.npz (default: {_DEFAULT_NPZ})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASKS_WEBB_JSON,
        help=f"Output JSON path (default: {TASKS_WEBB_JSON})",
    )
    args = parser.parse_args()
    path = write_tasks_webb_json(output=args.output, npz_path=args.npz)
    data = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"Wrote {len(data['tasks'])} tasks and "
        f"{len(data['icl'])} ICL types → {path}"
    )


if __name__ == "__main__":
    main()
