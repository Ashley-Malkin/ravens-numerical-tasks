"""Parse model output into option index 0–3 for ravens-numerical-tasks."""

from __future__ import annotations

import json
import re
from typing import Optional


def parse_structured_choice(response: str) -> Optional[int]:
    """Parse JSON with key \"choice\" (value A/B/C/D) if present (incl. ```json fences)."""
    try:
        stripped = response.strip()
        if "```" in stripped:
            m = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", stripped, re.DOTALL | re.I)
            if m:
                stripped = m.group(1).strip()
        if "{" not in stripped:
            return None
        start = stripped.index("{")
        depth = 0
        end = start
        for j, ch in enumerate(stripped[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        obj = json.loads(stripped[start:end])
        if isinstance(obj, dict) and "choice" in obj:
            c = str(obj["choice"]).strip().upper()
            if c and c[0] in "ABCD":
                return ord(c[0]) - ord("A")
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError):
        pass
    return None


def parse_answer(
    response: str,
    answer_options: Optional[list] = None,
    correct_index: Optional[int] = None,
) -> Optional[int]:
    """Extract option index 0-3 from model response. Accepts JSON choice, A/B/C/D, \\boxed{A},
    0-3, 1-4 (1-indexed), or the correct value (with word boundary).
    """
    sj = parse_structured_choice(response)
    if sj is not None:
        return sj

    resp = response.strip().upper()

    boxed_letter = re.search(r"\\boxed\{([A-D])\}", response, re.I)
    if boxed_letter:
        return ord(boxed_letter.group(1).upper()) - ord("A")
    boxed_num = re.search(r"\\boxed\{([0-3])\}", response)
    if boxed_num:
        return int(boxed_num.group(1))
    option_letter = list(re.finditer(r"\boption\s+([A-D])\b", response, re.I))
    if option_letter:
        return ord(option_letter[-1].group(1).upper()) - ord("A")
    option_num = list(re.finditer(r"\boption\s+([0-3])\b", response, re.I))
    if option_num:
        return int(option_num[-1].group(1))
    for m in re.finditer(
        r"(?:answer is|value is|blank is|option is|thus,? the value is)\s+([A-D])\b",
        response,
        re.I,
    ):
        return ord(m.group(1).upper()) - ord("A")
    last_answer = None
    for m in re.finditer(
        r"(?:answer is|value is|blank is|value for the blank is|thus,? the value is)\s+([0-4])\b",
        response,
        re.I,
    ):
        last_answer = m
    if last_answer:
        v = int(last_answer.group(1))
        return 0 if v == 0 else v - 1

    letter_matches = list(re.finditer(r"\b([A-D])\b", resp))
    if letter_matches:
        letter = letter_matches[-1].group(1)
        return ord(letter) - ord("A")

    for m in reversed(list(re.finditer(r"\b([0-3])\b", response))):
        return int(m.group(1))

    for m in reversed(list(re.finditer(r"(?:option\s*)?([1-4])\b", response, re.I))):
        return int(m.group(1)) - 1

    if answer_options is not None and correct_index is not None and 0 <= correct_index < 4:
        correct_val = answer_options[correct_index]
        if isinstance(correct_val, list):
            if len(correct_val) == 1:
                if re.search(r"\b" + re.escape(str(correct_val[0])) + r"\b", response):
                    return correct_index
            else:
                val_str = ",".join(str(x) for x in correct_val)
                if re.search(r"\b" + re.escape(val_str) + r"\b", response):
                    return correct_index
        else:
            if re.search(r"\b" + re.escape(str(correct_val)) + r"\b", response):
                return correct_index

    return None
