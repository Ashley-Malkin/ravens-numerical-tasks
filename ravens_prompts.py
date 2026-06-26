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


# Three fixed demonstrations per task_type (generated via ``generator.generate_in_context_examples``;
# same rules as ``tasks.json`` but disjoint matrices and full task fingerprints).
IN_CONTEXT_EXAMPLES: dict[str, list[IclExample]] = {
    "constancy": [
        {
            "matrix": [[11, 11, 11], [11, 11, 11], [11, 11, None]],
            "answer_options": [13, 12, 10, 11],
            "correct_letter": "D",
        },
        {
            "matrix": [[6, 6, 6], [6, 6, 6], [6, 6, None]],
            "answer_options": [4, 5, 6, 7],
            "correct_letter": "C",
        },
        {
            "matrix": [[5, 5, 5], [5, 5, 5], [5, 5, None]],
            "answer_options": [6, 5, 7, 4],
            "correct_letter": "B",
        },
    ],
    "pattern": [
        {
            "matrix": [[8, 11, 11], [8, 11, 11], [8, 11, None]],
            "answer_options": [13, 9, 11, 12],
            "correct_letter": "C",
        },
        {
            "matrix": [[18, 4, 4], [18, 4, 4], [18, 4, None]],
            "answer_options": [5, 2, 4, 6],
            "correct_letter": "C",
        },
        {
            "matrix": [[12, 3, 3], [12, 3, 3], [12, 3, None]],
            "answer_options": [5, 2, 3, 1],
            "correct_letter": "C",
        },
    ],
    "pattern_tuple": [
        {
            "matrix": [[[8], [4, 3], [4, 3]], [[4, 3], [8], [4, 3]], [[4, 3], [4, 3], None]],
            "answer_options": [[8], [3], [1, 0], [1, 4]],
            "correct_letter": "A",
        },
        {
            "matrix": [[[2], [9, 7], [9, 7]], [[9, 7], [2], [9, 7]], [[9, 7], [9, 7], None]],
            "answer_options": [[1, 2], [9], [8], [2]],
            "correct_letter": "D",
        },
        {
            "matrix": [[[9], [3, 8], [3, 8]], [[3, 8], [9], [3, 8]], [[3, 8], [3, 8], None]],
            "answer_options": [[4], [9], [1, 1], [1, 4]],
            "correct_letter": "B",
        },
    ],
    "progression": [
        {
            "matrix": [[18, 19, 20], [10, 11, 12], [10, 11, None]],
            "answer_options": [11, 14, 13, 12],
            "correct_letter": "D",
        },
        {
            "matrix": [[5, 6, 7], [4, 5, 6], [4, 5, None]],
            "answer_options": [6, 7, 8, 5],
            "correct_letter": "A",
        },
        {
            "matrix": [[18, 19, 20], [15, 16, 17], [15, 16, None]],
            "answer_options": [17, 18, 16, 15],
            "correct_letter": "A",
        },
    ],
    "combine": [
        {
            "matrix": [[[8], [6], [8, 6]], [[1], [5], [1, 5]], [[1], [8], None]],
            "answer_options": [[1, 5], [1, 8], [7, 1], [5, 3]],
            "correct_letter": "B",
        },
        {
            "matrix": [[[4], [6], [4, 6]], [[3], [4], [3, 4]], [[8], [6], None]],
            "answer_options": [[8, 6], [8, 7], [5, 9], [1, 6]],
            "correct_letter": "A",
        },
        {
            "matrix": [[[6], [2], [6, 2]], [[8], [1], [8, 1]], [[5], [4], None]],
            "answer_options": [[6, 6], [4, 3], [5, 4], [9, 3]],
            "correct_letter": "C",
        },
    ],
    "intersection": [
        {
            "matrix": [[[2, 3], [2, 8], [2]], [[4, 0], [7, 0], [0]], [[8, 4], [5, 4], None]],
            "answer_options": [2, 6, 4, 5],
            "correct_letter": "C",
        },
        {
            "matrix": [[[7, 5], [3, 7], [7]], [[3, 8], [8, 0], [8]], [[6, 1], [6, 2], None]],
            "answer_options": [5, 6, 8, 4],
            "correct_letter": "B",
        },
        {
            "matrix": [[[9, 3], [6, 3], [3]], [[8, 6], [7, 6], [6]], [[7, 3], [7, 9], None]],
            "answer_options": [9, 5, 7, 8],
            "correct_letter": "C",
        },
    ],
}


def _resolve_task_type(task_type: str) -> str:
    return task_type if task_type in IN_CONTEXT_EXAMPLES else "constancy"


def _format_one_example(
    ex: IclExample,
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
        _format_one_example(ex, mode, example_index=i + 1)
        for i, ex in enumerate(chosen)
    ]
    return "".join(blocks)


def build_prompt(
    task: dict,
    mode: str = "plain",
    n_examples: int = 0,
    prompt_type: str = "instruction",
) -> str:
    """Build the prompt for one task.

    ``prompt_type`` is ``instruction`` (letter MCQ; default) or ``completion`` (bracket fill-in).
    ``mode`` applies to ``instruction`` only (``plain``, ``choice_only``, ``cot_choice``).
    ``n_examples``: ``0`` = zero-shot; ``1`` / ``3`` = that many ICL demos from
    ``IN_CONTEXT_EXAMPLES`` (capped at three per task type).
    """
    if prompt_type == "completion":
        return build_completion_prompt(task, n_examples=n_examples)

    matrix = task["matrix"]
    options = task["answer_options"]
    task_type = task["task_type"]

    icl = in_context_example_block(task_type, mode, n_examples=n_examples)
    icl_section = icl if icl else ""

    common = f"""You are solving a numerical/spatial reasoning task (Raven's-style matrix). The blank is shown as ?. One of the options below is the correct answer.

{icl_section}Now solve this task.

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


# --- Completion-style prompts (baby-reasoning / matrix_easy bracket format) ---

COMPLETION_PERM_INVARIANT_TYPES = frozenset({"combine", "intersection"})


def format_completion_cell(cell: Any) -> str:
    """Format a matrix cell for bracket completion (space-separated values)."""
    if cell is None:
        return ""
    if isinstance(cell, list):
        return " ".join(str(v) for v in cell)
    return str(cell)


def format_completion_answer(option: Any) -> str:
    """Format an answer option for completion scoring."""
    if isinstance(option, list):
        return " ".join(str(v) for v in option)
    return str(option)


def matrix_to_completion_query(matrix: list) -> str:
    """3×3 matrix as bracketed grid; bottom-right cell is open ``[`` (Webb / matrix_easy style)."""
    rows = []
    for r in range(3):
        cells = []
        for c in range(3):
            if r == 2 and c == 2:
                cells.append("[")
            else:
                cells.append("[" + format_completion_cell(matrix[r][c]) + "]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def expected_completion_answer(task: dict) -> str:
    """Correct blank cell text from ``answer_options`` / ``correct_index``."""
    opts = task["answer_options"]
    ci = int(task["correct_index"])
    return format_completion_answer(opts[ci])


def completion_perm_invariant(task_type: str) -> bool:
    return task_type in COMPLETION_PERM_INVARIANT_TYPES


def _completion_icl_pairs(task_type: str, n_examples: int) -> list[tuple[str, str]]:
    if n_examples <= 0:
        return []
    tt = _resolve_task_type(task_type)
    pairs: list[tuple[str, str]] = []
    for ex in IN_CONTEXT_EXAMPLES[tt][: min(n_examples, len(IN_CONTEXT_EXAMPLES[tt]))]:
        query = matrix_to_completion_query(ex["matrix"])
        letter = ex["correct_letter"]
        idx = ord(str(letter).strip().upper()[0]) - ord("A")
        answer = format_completion_answer(ex["answer_options"][idx])
        pairs.append((query, answer))
    return pairs


def build_completion_prompt(task: dict, n_examples: int = 0) -> str:
    """Bracket-completion prompt on the same ``tasks.json`` items (no letter instructions)."""
    parts: list[str] = []
    for query, answer in _completion_icl_pairs(task["task_type"], n_examples):
        parts.append(query + answer + "]")
        parts.append("")
    parts.append(matrix_to_completion_query(task["matrix"]))
    return "\n".join(parts)
