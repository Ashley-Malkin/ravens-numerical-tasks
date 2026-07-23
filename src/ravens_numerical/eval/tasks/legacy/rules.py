from __future__ import annotations

import json
import random
from pathlib import Path

from ravens_numerical import LEGACY_DATA_DIR as DATA_DIR
from ravens_numerical.eval.tasks.base import ModelResponse, Stimulus, Task
from ravens_numerical.paths import TASKS_ABA_JSON

_SYLLABLES = [
    "ga",
    "ti",
    "li",
    "na",
    "ta",
    "da",
    "wo",
    "fe",
    "de",
    "ro",
    "ba",
    "fo",
    "bi",
    "ku",
    "me",
    "si",
    "pe",
    "zo",
    "re",
    "vi",
]

_RULES = ("ABA", "ABB")
_LEGACY_DATA_PATH = DATA_DIR / "rules" / "canonical.json"
_SECTION = "rules"
_ICL_SECTION = "rules_icl"
EASY_N_EXAMPLES = (0, 5, 10, 15, 20)


def _make_triplet(a: str, b: str, rule: str) -> tuple[str, str, str]:
    """Return (syllable_a, syllable_b, expected_third) for a given rule."""
    if rule == "ABA":
        return a, b, a
    else:  # ABB
        return a, b, b


def _load_easy_payload(tasks_json: Path | None) -> dict | None:
    path = tasks_json if tasks_json is not None else TASKS_ABA_JSON
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _load_rules_items(tasks_json: Path | None) -> list[dict]:
    """Load rules eval stimuli from ``tasks_aba.json`` or legacy canonical JSON."""
    data = _load_easy_payload(tasks_json)
    if data is not None and _SECTION in data and isinstance(data[_SECTION], list):
        return data[_SECTION]
    with open(_LEGACY_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_rules_icl(tasks_json: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Load shared per-rule ICL pools (up to 20 demos each)."""
    data = _load_easy_payload(tasks_json)
    out: dict[str, list[tuple[str, str]]] = {rule: [] for rule in _RULES}
    if data is None:
        return out
    raw = data.get(_ICL_SECTION, {})
    if not isinstance(raw, dict):
        return out
    for rule in _RULES:
        items = raw.get(rule, [])
        if isinstance(items, list):
            out[rule] = [tuple(e) for e in items if isinstance(e, (list, tuple)) and len(e) == 2]
    return out


def _icl_for_rule(
    rule: str,
    icl_pools: dict[str, list[tuple[str, str]]],
    n_examples: int,
) -> list[tuple[str, str]]:
    if n_examples <= 0:
        return []
    pool = icl_pools.get(rule, [])
    return list(pool[: min(n_examples, len(pool))])


class RulesTask(Task):
    """Marcus (1999) abstract rule learning over nonsense syllable triplets.

    Stimuli are formatted as sequence completions: two syllables shown,
    model predicts the third. No blank placeholders. In few-shot condition,
    complete triplets are given as context lines (base-model ICL style).

    Canonical eval items and ICL demos are loaded from ``tasks_aba.json``;
    ICL pools are shared per rule and disjoint from the eval set.
    """

    def __init__(self, tasks_json: Path | None = None) -> None:
        self._tasks_json = tasks_json
        self._icl_pools = _load_rules_icl(tasks_json)

    def canonical_stimuli(self) -> list[Stimulus]:
        items = _load_rules_items(self._tasks_json)
        stimuli = []
        for item in items:
            rule = str(item.get("metadata", {}).get("rule", "ABB"))
            # Attach full shared ICL pool; build_prompt slices by n_examples.
            few_shot = list(self._icl_pools.get(rule, []))
            if not few_shot:
                few_shot = [tuple(e) for e in item.get("few_shot_examples", [])]
            stimuli.append(
                Stimulus(
                    query=item["query"],
                    expected=item["expected"],
                    few_shot_examples=few_shot,
                    metadata=item.get("metadata", {}),
                    answer_choices=item.get("answer_choices"),
                )
            )
        return stimuli

    def generate_stimulus(self, n_examples: int = 3) -> Stimulus:
        rule = random.choice(_RULES)
        pool = _SYLLABLES.copy()
        random.shuffle(pool)
        a, b, expected = _make_triplet(pool[0], pool[1], rule)

        examples = _icl_for_rule(rule, self._icl_pools, n_examples)
        if n_examples > 0 and not examples:
            for _ in range(n_examples):
                ex_pool = _SYLLABLES.copy()
                random.shuffle(ex_pool)
                ea, eb, ex_ans = _make_triplet(ex_pool[0], ex_pool[1], rule)
                examples.append((f"{ea} {eb}", ex_ans))

        return Stimulus(
            query=f"{a} {b}",
            expected=expected,
            few_shot_examples=examples,
            metadata={"rule": rule, "source": "generated"},
            answer_choices=[a, b],
        )

    def systematic_stimuli(self, n_per_rule: int, n_examples: int) -> list[Stimulus]:
        """Generate stimuli covering each rule with ``n_per_rule`` instances."""
        stimuli = []
        for rule in _RULES:
            for _ in range(n_per_rule):
                pool = _SYLLABLES.copy()
                random.shuffle(pool)
                a, b, expected = _make_triplet(pool[0], pool[1], rule)
                examples = _icl_for_rule(rule, self._icl_pools, n_examples)
                if n_examples > 0 and not examples:
                    for _ in range(n_examples):
                        ex_pool = _SYLLABLES.copy()
                        random.shuffle(ex_pool)
                        ea, eb, ex_ans = _make_triplet(ex_pool[0], ex_pool[1], rule)
                        examples.append((f"{ea} {eb}", ex_ans))
                stimuli.append(
                    Stimulus(
                        query=f"{a} {b}",
                        expected=expected,
                        few_shot_examples=examples,
                        metadata={"rule": rule, "source": "systematic"},
                        answer_choices=[a, b],
                    )
                )
        return stimuli

    def format_completion(self, stimulus: Stimulus, choice: str) -> str:
        return " " + choice

    def score(self, response: ModelResponse, stimulus: Stimulus) -> bool:
        text = response.text.strip()
        expected = stimulus.expected.strip().lower()
        if not text:
            return False
        if text.lower() == expected:
            return True
        # First whitespace-separated token (handles continued generation).
        return text.split()[0].lower() == expected

    def build_prompt(self, stimulus: Stimulus, n_examples: int) -> str:
        if n_examples > 0 and stimulus.few_shot_examples:
            examples = stimulus.few_shot_examples[:n_examples]
            lines = [f"{q} {a}" for q, a in examples]
            lines.append(stimulus.query)
            return "\n".join(lines)
        return stimulus.query
