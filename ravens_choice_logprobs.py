"""Choice-letter logprob parsing and Brier metrics for ravens instruction (choice-only) eval."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

# JSON schema for forced choice output (Ollama / vLLM structured generation).
CHOICE_ONLY_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        }
    },
    "required": ["choice"],
}

# CoT + forced choice: reasoning string then same choice enum; logprobs align to choice letter token.
COT_CHOICE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "choice": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        },
    },
    "required": ["reasoning", "choice"],
}


def read_logprob(obj: dict) -> Optional[float]:
    """Read logprob from a logprobs entry (``logprob`` / ``log_prob``). Treat (0,1] as probability."""
    raw = obj.get("logprob") if obj.get("logprob") is not None else obj.get("log_prob")
    if raw is None:
        return None
    v = float(raw)
    if v == 0:
        return None
    if 0 < v <= 1:
        return math.log(v)
    return v


def letter_logprobs_single_entry(entry: dict) -> dict[str, float]:
    """Collect best logprob per letter A–D from one logprob step (chosen token + ``top_logprobs``)."""
    letters = "ABCD"
    best: dict[str, float] = {c: float("-inf") for c in letters}
    token_str = (entry.get("token") or "").strip().upper()
    if token_str and token_str[0] in letters:
        lp = read_logprob(entry)
        if lp is not None:
            best[token_str[0]] = max(best[token_str[0]], lp)
    for alt in entry.get("top_logprobs") or []:
        t = (alt.get("token") or "").strip().upper()
        if t and t[0] in letters:
            lp = read_logprob(alt)
            if lp is not None:
                best[t[0]] = max(best[t[0]], lp)
    return best


def choice_letter_char_index(text: str) -> Optional[int]:
    """Start index of the A–D value in ``"choice":"X"`` (0-based in ``text``)."""
    m = re.search(r'(?i)"choice"\s*:\s*"\s*([ABCD])', text.strip())
    if m:
        return m.start(1)
    return None


def char_index_to_token_index(tokens: list[str], char_pos: int) -> Optional[int]:
    """Which generated token covers ``char_pos`` in ``"".join(tokens)``."""
    if char_pos < 0:
        return None
    off = 0
    for i, t in enumerate(tokens):
        ln = len(t)
        if off <= char_pos < off + ln:
            return i
        off += ln
    return None


def parse_logprobs_by_letter(
    resp_data: dict,
    response_text: str,
) -> tuple[Optional[int], dict[str, float]]:
    """Logprobs at the token containing the JSON choice letter (Ollama ``logprobs`` list shape)."""
    letters = "ABCD"
    empty = {c: float("-inf") for c in letters}
    logprobs_list = resp_data.get("logprobs") or []
    if not logprobs_list:
        return None, empty

    tokens = [e.get("token") or "" for e in logprobs_list]
    full = "".join(tokens)

    def pos_in_full() -> Optional[int]:
        p = choice_letter_char_index(full)
        if p is not None:
            return p
        mr = re.search(r'(?i)"choice"\s*:\s*"\s*([ABCD])', response_text.strip())
        if not mr:
            return None
        letter = mr.group(1).upper()
        mfull = re.search(r'(?i)("choice"\s*:\s*"\s*)(' + letter + ")", full)
        if mfull:
            return mfull.start(2)
        return None

    pos = pos_in_full()
    if pos is None:
        return None, empty

    ti = char_index_to_token_index(tokens, pos)
    if ti is None:
        return None, empty

    best = letter_logprobs_single_entry(logprobs_list[ti])
    top = max(letters, key=lambda c: best[c])
    if best[top] == float("-inf"):
        return None, best
    return ord(top) - ord("A"), best


def vllm_logprobs_to_ollama_list(resp_data: dict) -> list[dict]:
    """Normalize vLLM OpenAI chat or completions logprobs to Ollama-style step list."""
    choice = (resp_data.get("choices") or [{}])[0]
    logprobs_data = choice.get("logprobs") or {}

    if isinstance(logprobs_data, dict) and "content" in logprobs_data:
        entries: list[dict] = []
        for step in logprobs_data["content"] or []:
            entries.append(
                {
                    "token": step.get("token") or "",
                    "logprob": step.get("logprob"),
                    "top_logprobs": [
                        {"token": t.get("token"), "logprob": t.get("logprob")}
                        for t in (step.get("top_logprobs") or [])
                    ],
                }
            )
        return entries

    tokens = logprobs_data.get("tokens") or []
    token_logprobs = logprobs_data.get("token_logprobs") or []
    top_logprobs = logprobs_data.get("top_logprobs") or []
    entries = []
    for i, token in enumerate(tokens):
        lp = token_logprobs[i] if i < len(token_logprobs) else None
        entry: dict = {"token": token, "logprob": lp, "top_logprobs": []}
        tops = top_logprobs[i] if i < len(top_logprobs) else None
        if isinstance(tops, dict):
            entry["top_logprobs"] = [
                {"token": k, "logprob": v} for k, v in tops.items()
            ]
        elif isinstance(tops, list):
            entry["top_logprobs"] = tops
        entries.append(entry)
    return entries


def parse_logprobs_by_letter_vllm(
    resp_data: dict,
    response_text: str,
) -> tuple[Optional[int], dict[str, float]]:
    """Choice-token logprobs from a vLLM OpenAI API response."""
    entries = vllm_logprobs_to_ollama_list(resp_data)
    return parse_logprobs_by_letter({"logprobs": entries}, response_text)


def has_finite_letter_logprob(best: dict[str, float]) -> bool:
    """True if at least one A–D has a finite logprob."""
    return any(v > float("-inf") and not math.isnan(v) for v in best.values())


def letter_logprobs_to_probs(best: dict[str, float]) -> Optional[list[float]]:
    """Softmax over A,B,C,D into probabilities (order A..D). Missing letters use a log floor."""
    letters = "ABCD"
    log_floor = -1e9
    lps: list[float] = []
    for c in letters:
        v = best.get(c, float("-inf"))
        if v == float("-inf") or math.isnan(v):
            v = log_floor
        lps.append(v)
    m = max(lps)
    exps = [math.exp(x - m) for x in lps]
    s = sum(exps)
    if s <= 0:
        return None
    return [e / s for e in exps]


def multiclass_brier_score(probs: list[float], correct_idx: int) -> float:
    """Multiclass Brier: ``(1/K) * sum_k (p_k - o_k)^2``. Lower is better."""
    k = len(probs)
    if not (0 <= correct_idx < k):
        raise ValueError("correct_idx out of range")
    o = [0.0] * k
    o[correct_idx] = 1.0
    return sum((probs[i] - o[i]) ** 2 for i in range(k)) / k
