"""Helpers for vLLM echo completion logprob scoring."""

from __future__ import annotations

# Near-zero sums (all options ≈ 0) are an echo artifact, not a real ranking.
DEFAULT_MIN_SPREAD = 1e-3
DEFAULT_ABS_TOL = 1e-4


def completion_option_length(choice: str) -> int:
    """Whitespace-delimited pieces in a completion option (``"10 11]"`` → 2)."""
    body = choice.replace("]", "").strip()
    return max(len(body.split()), 1)


def sum_echo_completion_logprobs(
    token_logprobs: list[float | None],
    text_offset: list[int],
    prompt_len: int,
    *,
    full_len: int | None = None,
) -> float | None:
    """Sum token logprobs that overlap the completion character span.

    Tokens are treated as half-open spans ``[text_offset[i], text_offset[i+1])``
    (last token ends at ``full_len``). A token is included if it overlaps
    ``[prompt_len, full_len)``, so BPE merges that straddle the prompt/completion
    boundary (e.g. ``[`` + ``185``) still contribute.

    Returns ``None`` when no overlapping tokens have a finite logprob (empty
    match). Distinguishes that failure from a real sum of ``0.0``.
    """
    if len(token_logprobs) != len(text_offset):
        raise ValueError(
            f"token_logprobs length {len(token_logprobs)} != "
            f"text_offset length {len(text_offset)}"
        )
    if not text_offset:
        return None

    if full_len is not None:
        end = full_len
    else:
        end = max(text_offset[-1] + 1, prompt_len)

    completion_start = prompt_len
    completion_end = end
    if completion_end <= completion_start:
        return None

    matched: list[float] = []
    n = len(text_offset)
    for i, (lp, off) in enumerate(zip(token_logprobs, text_offset)):
        tok_start = off
        tok_end = text_offset[i + 1] if i + 1 < n else end
        if tok_end <= tok_start:
            tok_end = tok_start + 1
        # Overlap with [completion_start, completion_end)
        if tok_end > completion_start and tok_start < completion_end:
            if lp is not None:
                matched.append(float(lp))

    if not matched:
        return None
    return sum(matched)


def unique_logprob_argmax(
    logprobs: dict[str, float],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    min_spread: float = DEFAULT_MIN_SPREAD,
    length_normalize: bool = True,
) -> str | None:
    """Return the unique max-logprob key, or ``None`` if empty, degenerate, or tied.

    When ``length_normalize`` is true, compares mean logprob per whitespace
    token so a short distractor like ``"10"`` is not preferred to ``"10 11"``
    solely because the sum includes fewer tokens.

    If the range of (normalized) scores is below ``min_spread``, the ranking is
    treated as collapsed (typical when echo returns ≈0 for every option) and
    this returns ``None`` instead of an arbitrary float-noise winner.
    """
    if not logprobs:
        return None
    scores = dict(logprobs)
    if length_normalize:
        scores = {
            key: value / completion_option_length(key)
            for key, value in logprobs.items()
        }
    values = list(scores.values())
    if max(values) - min(values) < min_spread:
        return None
    max_lp = max(values)
    winners = [key for key, lp in scores.items() if max_lp - lp <= abs_tol]
    if len(winners) != 1:
        return None
    return winners[0]
