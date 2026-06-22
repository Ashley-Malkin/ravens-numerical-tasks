"""Shared prompt construction for Raven's numerical tasks (plain / JSON modes).

Used by evaluate.py and baby_reasoning RavensNumericalTask.
"""

from __future__ import annotations

import json
from typing import Any

IclExample = dict[str, Any]


def format_cell(cell: Any) -> str:
    """Format a matrix cell for display (number, null, or tuple)."""
    if cell is None:
        return "?"
    if isinstance(cell, list):
        if len(cell) == 1:
            return str(cell[0])
        return "(" + ",".join(str(x) for x in cell) + ")"
    return str(cell)


def format_matrix(matrix: list) -> str:
    """Format 2x2 or 3x3 matrix as a readable grid."""
    rows = []
    for row in matrix:
        cells = [format_cell(c) for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def format_options(options: list) -> str:
    """Format answer options as A/B/C/D choices."""
    letters = "ABCD"
    lines = []
    for i, opt in enumerate(options):
        letter = letters[i] if i < 4 else str(i)
        if isinstance(opt, list):
            if len(opt) == 1:
                lines.append(f"  {letter}: {opt[0]}")
            else:
                lines.append(f"  {letter}: ({','.join(str(x) for x in opt)})")
        else:
            lines.append(f"  {letter}: {opt}")
    return "\n".join(lines)


# Three fixed demonstrations per task_type (from tasks.json; letters match option order).
IN_CONTEXT_EXAMPLES: dict[str, list[IclExample]] = {
    "constancy": [
        {
            "matrix": [[3, 3, 3], [3, 3, 3], [3, 3, None]],
            "answer_options": [4, 3, 1, 2],
            "correct_letter": "B",
        },
        {
            "matrix": [[8, 8, 8], [8, 8, 8], [8, 8, None]],
            "answer_options": [8, 9, 10, 7],
            "correct_letter": "A",
        },
        {
            "matrix": [[12, 12, 12], [12, 12, 12], [12, 12, None]],
            "answer_options": [14, 10, 13, 12],
            "correct_letter": "D",
        },
    ],
    "pattern": [
        {
            "matrix": [[19, 15, 15], [19, 15, 15], [19, 15, None]],
            "answer_options": [17, 15, 13, 14],
            "correct_letter": "B",
        },
        {
            "matrix": [[19, 12, 12], [19, 12, 12], [19, 12, None]],
            "answer_options": [13, 12, 14, 10],
            "correct_letter": "B",
        },
        {
            "matrix": [[18, 2, 2], [18, 2, 2], [18, 2, None]],
            "answer_options": [2, 1, 4, 3],
            "correct_letter": "A",
        },
    ],
    "pattern_tuple": [
        {
            "matrix": [
                [[5], [9, 2], [9, 2]],
                [[9, 2], [5], [9, 2]],
                [[9, 2], [9, 2], None],
            ],
            "answer_options": [[4], [1, 4], [5], [8]],
            "correct_letter": "C",
        },
        {
            "matrix": [
                [[1], [9, 3], [9, 3]],
                [[9, 3], [1], [9, 3]],
                [[9, 3], [9, 3], None],
            ],
            "answer_options": [[1], [0], [1, 3], [1, 4]],
            "correct_letter": "A",
        },
        {
            "matrix": [
                [[7], [2, 0], [2, 0]],
                [[2, 0], [7], [2, 0]],
                [[2, 0], [2, 0], None],
            ],
            "answer_options": [[7], [1, 0], [6], [1, 1]],
            "correct_letter": "A",
        },
    ],
    "progression": [
        {
            "matrix": [[13, 14, 15], [11, 12, 13], [11, 12, None]],
            "answer_options": [12, 15, 13, 11],
            "correct_letter": "C",
        },
        {
            "matrix": [[4, 5, 6], [14, 15, 16], [14, 15, None]],
            "answer_options": [17, 14, 16, 15],
            "correct_letter": "C",
        },
        {
            "matrix": [[5, 6, 7], [18, 19, 20], [18, 19, None]],
            "answer_options": [17, 18, 19, 20],
            "correct_letter": "D",
        },
    ],
    "combine": [
        {
            "matrix": [
                [[5], [9], [5, 9]],
                [[5], [4], [5, 4]],
                [[4], [5], None],
            ],
            "answer_options": [[2, 8], [6, 4], [4, 5], [9, 3]],
            "correct_letter": "C",
        },
        {
            "matrix": [
                [[3], [4], [3, 4]],
                [[4], [2], [4, 2]],
                [[2], [7], None],
            ],
            "answer_options": [[2, 7], [1, 7], [2, 1], [4, 1]],
            "correct_letter": "A",
        },
        {
            "matrix": [
                [[6], [1], [6, 1]],
                [[4], [7], [4, 7]],
                [[3], [6], None],
            ],
            "answer_options": [[3, 6], [8, 4], [3, 7], [8, 5]],
            "correct_letter": "A",
        },
    ],
    "intersection": [
        {
            "matrix": [
                [[5, 0], [9, 0], [0]],
                [[8, 7], [0, 8], [8]],
                [[0, 7], [4, 7], None],
            ],
            "answer_options": [6, 8, 7, 5],
            "correct_letter": "C",
        },
        {
            "matrix": [
                [[1, 8], [2, 1], [1]],
                [[5, 9], [9, 6], [9]],
                [[3, 0], [3, 8], None],
            ],
            "answer_options": [3, 1, 4, 5],
            "correct_letter": "A",
        },
        {
            "matrix": [
                [[3, 1], [1, 7], [1]],
                [[0, 3], [3, 5], [3]],
                [[9, 7], [6, 9], None],
            ],
            "answer_options": [8, 9, 6, 7],
            "correct_letter": "B",
        },
    ],
}


def _resolve_task_type(task_type: str) -> str:
    return task_type if task_type in IN_CONTEXT_EXAMPLES else "constancy"


def _format_one_example(
    ex: IclExample,
    task_type: str,
    mode: str,
    *,
    example_index: int | None = None,
) -> str:
    letter = ex["correct_letter"]
    matrix_s = format_matrix(ex["matrix"])
    opts_s = format_options(ex["answer_options"])
    label = (
        f"Example {example_index} (illustration only; your task below is different)"
        if example_index is not None
        else "Example (illustration only; your task below is different)"
    )
    if mode == "choice_only":
        correct_json = json.dumps({"choice": letter}, separators=(",", ":"))
        footer = (
            f"For this example, the correct option letter is {letter}. "
            f"A valid reply would be exactly: {correct_json}"
        )
    elif mode == "cot_choice":
        example_obj = {"reasoning": "", "choice": letter}
        correct_json = json.dumps(example_obj, separators=(",", ":"))
        footer = (
            f"For this example, the correct option letter is {letter}. "
            f"Required JSON shape (put your reasoning in the string; here it is empty "
            f"for illustration): {correct_json}"
        )
    else:
        footer = f"For this example, the correct response is the option letter {letter}."

    return f"""{label}:

Task type: {task_type}

Matrix:
{matrix_s}

Options (each has a letter A, B, C, or D):
{opts_s}

{footer}

"""


def in_context_example_block(
    task_type: str,
    mode: str,
    n_examples: int = 0,
) -> str:
    """Format in-context demonstration(s) for ``task_type``.

    ``mode`` is ``plain``, ``choice_only``, or ``cot_choice``.
    ``n_examples``: ``0`` = no demos; ``1`` = first demo; ``3`` = all three stored demos.
    """
    if n_examples <= 0:
        return ""

    tt = _resolve_task_type(task_type)
    examples = IN_CONTEXT_EXAMPLES[tt]
    chosen = examples[: min(n_examples, len(examples))]

    blocks = [
        _format_one_example(ex, tt, mode, example_index=i + 1)
        for i, ex in enumerate(chosen)
    ]
    return "".join(blocks)


def build_prompt(
    task: dict,
    mode: str = "plain",
    n_examples: int = 0,
) -> str:
    """Build the prompt for one task.

    ``mode`` is ``plain`` (letter only), ``choice_only`` (JSON choice only), or
    ``cot_choice`` (JSON with reasoning + choice).
    ``n_examples``: ``0`` = zero-shot; ``1`` / ``3`` = that many ICL demos from
    ``IN_CONTEXT_EXAMPLES`` (capped at three per task type).
    """
    matrix = task["matrix"]
    options = task["answer_options"]
    task_type = task["task_type"]

    icl = in_context_example_block(task_type, mode, n_examples=n_examples)
    icl_section = icl if icl else ""

    common = f"""You are solving a numerical/spatial reasoning task (Raven's-style matrix). The blank is shown as ?. One of the options below is the correct answer.

{icl_section}Now solve this task.

Task type: {task_type}

Matrix:
{format_matrix(matrix)}

Options (each has a letter A, B, C, or D):
{format_options(options)}
"""
    if mode == "choice_only":
        return (
            common
            + f"""
Your entire reply must be a single JSON object with exactly one key "choice" whose value is one of the strings "A", "B", "C", or "D" (the option letter only).
Example: {{"choice":"B"}}
Do not include any other text, keys, or explanation."""
        )
    if mode == "cot_choice":
        return (
            common
            + """
Your entire reply must be a single JSON object with exactly two keys:
  "reasoning": a string with your step-by-step reasoning;
  "choice": one of the strings "A", "B", "C", or "D" (the option letter only).
Example: {"reasoning":"","choice":"B"}
Put your step-by-step work in "reasoning" (start from empty content and fill it); "choice" is the option letter only.
Do not include any other keys or any text outside the JSON object."""
        )

    return (
        common
        + """
Reply with ONLY the option letter: A, B, C, or D. (A = first option, B = second, C = third, D = fourth.)
Do not give the value that goes in the blank—give the option letter.

Do not provide additional reasoning or explanation, only return the option letter."""
    )
