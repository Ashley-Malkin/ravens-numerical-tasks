"""Recompute forced-choice accuracy from exported trial JSON.

Uses the same rule as ``evaluate`` after the echo-argmax fix: unique
length-normalized echo logprob argmax, with generation match only when echo
abstains. Plot loaders call ``overlay_log_scores`` so markdown logs are replaced
when a matching dump exists.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ravens_numerical.paths import RUNS_DIR
from ravens_numerical.scoring.echo_logprobs import unique_logprob_argmax


@dataclass(frozen=True)
class DumpSummary:
    overall: float
    by_task: dict[str, float]
    n: int
    tag: str
    suite: str


def infer_suite(data: list[dict]) -> str:
    types = {
        row["stimulus"]["metadata"]["task"]["task_type"] for row in data
    }
    if "tuple_grid" in types:
        return "challenge"

    def _max_abs(obj: object, acc: int = 0) -> int:
        if isinstance(obj, bool):
            return acc
        if isinstance(obj, int):
            return max(acc, abs(obj))
        if isinstance(obj, list):
            for item in obj:
                acc = _max_abs(item, acc)
            return acc
        return acc

    opts = data[0]["stimulus"]["metadata"]["task"].get("answer_options")
    return "5digit" if _max_abs(opts) >= 10000 else "regular"


def _gen_match(row: dict) -> str | None:
    text = (row.get("response") or {}).get("text") or ""
    span = text.split("]")[0].strip()
    choices = (row.get("stimulus") or {}).get("answer_choices") or []
    choice_map = {c.lower(): c for c in choices}
    span_l = span.lower()
    if span_l in choice_map:
        return choice_map[span_l]
    bare = span_l.rstrip("]").strip()
    if bare in choice_map:
        return choice_map[bare]
    return None


def trial_correct(row: dict) -> bool:
    gold = row["stimulus"]["expected"]
    logprobs = {
        key: value
        for key, value in (row.get("score", {}).get("answer_logprobs") or {}).items()
        if value is not None
    }
    pred = unique_logprob_argmax(logprobs) if logprobs else None
    if pred is not None:
        return pred == gold
    matched = _gen_match(row)
    if matched is not None:
        return matched == gold
    return False


def summarize_trials(data: list[dict], *, tag: str, suite: str) -> DumpSummary:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in data:
        task = row["stimulus"]["metadata"]["task"]["task_type"]
        counts[task][1] += 1
        counts[task][0] += int(trial_correct(row))
    n = sum(total for _ok, total in counts.values())
    overall = (
        100.0 * sum(ok for ok, _total in counts.values()) / n if n else 0.0
    )
    by_task = {
        task: 100.0 * ok / total for task, (ok, total) in counts.items() if total
    }
    return DumpSummary(
        overall=overall, by_task=by_task, n=n, tag=tag, suite=suite
    )


def _tag_matches_query(tag: str, query: str) -> bool:
    """True if ``query`` (log model id or SFT run_id) names this dump folder."""
    tag_n = tag.replace("/", "--")
    q = query.strip().replace("/", "--").strip("-")
    if not q:
        return False
    if re.search(rf"(?:^|--){re.escape(q)}(?:--|__|$)", tag_n):
        return True
    q_parts = [part for part in q.split("--") if part]
    tag_parts = [part for part in tag_n.split("--") if part]
    idx = 0
    for part in tag_parts:
        if idx >= len(q_parts):
            break
        want = q_parts[idx]
        if part == want or part.startswith(want + "__"):
            idx += 1
    if idx == len(q_parts):
        return True
    last = q_parts[-1]
    if len(last) < 8 and "_" not in last:
        return False
    return any(part == last or part.startswith(last + "__") for part in tag_parts)


class DumpScoreIndex:
    def __init__(self, by_tag_suite: dict[tuple[str, str], DumpSummary]) -> None:
        self._by_tag_suite = by_tag_suite

    @classmethod
    def from_runs_dir(cls, runs_dir: Path | None = None) -> DumpScoreIndex:
        root = runs_dir or RUNS_DIR
        latest: dict[tuple[str, str], tuple[str, DumpSummary]] = {}
        if not root.is_dir():
            return cls({})
        for path in root.glob("**/ravens_numerical/0_examples_completion.json"):
            tag = path.parts[-4]
            stamp = path.parts[-3]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not data:
                continue
            try:
                suite = infer_suite(data)
            except (KeyError, TypeError):
                continue
            key = (tag, suite)
            prev = latest.get(key)
            if prev is not None and prev[0] >= stamp:
                continue
            latest[key] = (stamp, summarize_trials(data, tag=tag, suite=suite))
        return cls({key: summary for key, (_stamp, summary) in latest.items()})

    def lookup(self, query: str, suite: str) -> DumpSummary | None:
        hits = [
            summary
            for (tag, su), summary in self._by_tag_suite.items()
            if su == suite and _tag_matches_query(tag, query)
        ]
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        hits.sort(key=lambda item: len(item.tag))
        return hits[0]


@lru_cache(maxsize=1)
def default_index() -> DumpScoreIndex:
    return DumpScoreIndex.from_runs_dir()


def log_preamble_suite(text: str, match_start: int) -> str:
    """Infer eval suite from the nearest ``_Logged`` / ``_Generated`` line."""
    head = text[:match_start]
    meta = None
    for match in re.finditer(r"^_(?:Logged|Generated).+$", head, re.MULTILINE):
        meta = match.group(0)
    if meta is not None:
        if "challenge_tasks" in meta:
            return "challenge"
        if "tasks_5digit" in meta:
            return "5digit"
        return "regular"
    preamble = head[-2000:]
    if "challenge_tasks" in preamble or "tuple_grid" in preamble:
        return "challenge"
    if "tasks_5digit" in preamble:
        return "5digit"
    return "regular"


def overlay_regular_section(
    text: str,
    match_start: int,
    query: str,
    overall: float | None,
    by_task: dict[str, float],
) -> tuple[float | None, dict[str, float]] | None:
    """Overlay regular-suite dumps; skip 5-digit / challenge markdown.

    Combined ``*_sft_evals_n0.md`` files are often last-suite-only. Those
    matches return None here; callers should fill from dump_regular_scores.
    """
    suite = log_preamble_suite(text, match_start)
    if by_task and "tuple_grid" in by_task:
        suite = "challenge"
    if suite != "regular":
        return None
    return overlay_log_scores(query, "regular", overall, by_task)


def dump_regular_scores(
    query: str,
) -> tuple[float, dict[str, float]] | None:
    hit = default_index().lookup(query, "regular")
    if hit is None:
        return None
    return hit.overall, dict(hit.by_task)


def overlay_log_scores(
    query: str,
    suite: str,
    overall: float | None,
    by_task: dict[str, float],
) -> tuple[float | None, dict[str, float]]:
    """Replace markdown accuracies with dump-recomputed scores when available."""
    hit = default_index().lookup(query, suite)
    if hit is None:
        return overall, by_task
    return hit.overall, dict(hit.by_task)
