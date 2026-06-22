#!/usr/bin/env python3
"""Evaluate Qwen3 models on ravens-numerical-tasks via Ollama API (local or remote)."""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from ravens_answer_parse import parse_answer, parse_structured_choice
from ravens_prompts import (
    build_prompt,
    format_cell,
    format_matrix,
    format_options,
    in_context_example_block,
)

# When --debug, evaluate all of these models and report results for each.
# Ollama names use colons for tags. To download: ollama pull qwen3:30b && ollama pull qwen3:8b && ollama pull qwen3:4b
DEBUG_MODELS = ["qwen3:30b", "qwen3:8b", "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"]

# Ollama structured output for --choice-only (JSON Schema subset).
CHOICE_ONLY_FORMAT = {
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
COT_CHOICE_FORMAT = {
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


def _in_context_example_block(task_type: str, mode: str) -> str:
    """Alias for ``ravens_prompts.in_context_example_block`` (backward-compatible name)."""
    return in_context_example_block(task_type, mode)



def call_ollama(
    prompt: str,
    model: str = "qwen3:30b",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
    think: Optional[bool] = None,
    num_predict: Optional[int] = None,
    structured_format: Optional[dict] = None,
    logprobs: bool = False,
) -> tuple[str, dict]:
    """Send prompt to Ollama generate API. Returns ``(response_text, full_json)``.

    If ``structured_format`` is set (``--choice-only`` or ``--cot-choice``), sends Ollama ``format`` and ``think``: false.
    With ``logprobs`` (used for those modes), sets ``logprobs`` / ``top_logprobs`` on the request.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if structured_format is not None:
        payload["format"] = structured_format
        payload["think"] = False
    elif think is not None:
        payload["think"] = think
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    opts = payload.setdefault("options", {})
    opts["temperature"] = 0  
    if num_predict is not None:
        opts["num_predict"] = num_predict
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = (data.get("response") or "").strip()
    thinking_text = (data.get("thinking") or "").strip()
    # Many models (e.g. Qwen3) put the answer in "thinking"; with JSON format small models may only emit there.
    if not text and thinking_text:
        text = thinking_text
        print("Warning: empty 'response'; using 'thinking' field for parsing.", file=sys.stderr)
    elif not text:
        summary = {k: v for k, v in data.items() if k not in ("response", "thinking") and not (isinstance(v, list) and len(str(v)) > 200)}
        print(f"Warning: empty model response. API keys: {list(data.keys())}. Summary: {summary}", file=sys.stderr)
    return text, data


def _read_logprob(obj: dict) -> Optional[float]:
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


def _letter_logprobs_single_entry(entry: dict) -> dict[str, float]:
    """Collect best logprob per letter A–D from one logprob step (chosen token + ``top_logprobs``)."""
    letters = "ABCD"
    best: dict[str, float] = {c: float("-inf") for c in letters}
    token_str = (entry.get("token") or "").strip().upper()
    if token_str and token_str[0] in letters:
        lp = _read_logprob(entry)
        if lp is not None:
            best[token_str[0]] = max(best[token_str[0]], lp)
    for alt in entry.get("top_logprobs") or []:
        t = (alt.get("token") or "").strip().upper()
        if t and t[0] in letters:
            lp = _read_logprob(alt)
            if lp is not None:
                best[t[0]] = max(best[t[0]], lp)
    return best


def _choice_letter_char_index(text: str) -> Optional[int]:
    """Start index of the A–D value in ``"choice":"X"`` (0-based in ``text``)."""
    m = re.search(r'(?i)"choice"\s*:\s*"\s*([ABCD])', text.strip())
    if m:
        return m.start(1)
    return None


def _char_index_to_token_index(tokens: list[str], char_pos: int) -> Optional[int]:
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


def parse_logprobs_by_letter(resp_data: dict, response_text: str) -> tuple[Optional[int], dict[str, float]]:
    """Logprobs **only at the token that contains the JSON choice letter** (after ``"choice":``).

    Builds text as ``"".join`` of per-step ``token`` strings from ``logprobs`` (same order as
    generation), finds the ``A``–``D`` in ``"choice":"X"``, maps that character to its token
    index, then reads A–D masses from **that** step only (chosen token + ``top_logprobs``).

    If the regex does not match the concatenated tokens, tries again on ``response_text`` and
    maps the matched letter back onto ``full`` only when an identical ``"choice":"X"`` span exists
    there. Otherwise returns ``(None, {-inf,...})``.
    """
    letters = "ABCD"
    empty = {c: float("-inf") for c in letters}
    logprobs_list = resp_data.get("logprobs") or []
    if not logprobs_list:
        return None, empty

    tokens = [e.get("token") or "" for e in logprobs_list]
    full = "".join(tokens)

    def pos_in_full() -> Optional[int]:
        p = _choice_letter_char_index(full)
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

    ti = _char_index_to_token_index(tokens, pos)
    if ti is None:
        return None, empty

    best = _letter_logprobs_single_entry(logprobs_list[ti])
    top = max(letters, key=lambda c: best[c])
    if best[top] == float("-inf"):
        return None, best
    return ord(top) - ord("A"), best


def _has_finite_letter_logprob(best: dict[str, float]) -> bool:
    """True if at least one A–D has a finite logprob (partial ``top_logprobs`` is ok)."""
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
    """Multiclass Brier score: ``(1/K) * sum_k (p_k - o_k)^2`` with one-hot ``o``. Lower is better; 0 = perfect."""
    k = len(probs)
    if not (0 <= correct_idx < k):
        raise ValueError("correct_idx out of range")
    o = [0.0] * k
    o[correct_idx] = 1.0
    return sum((probs[i] - o[i]) ** 2 for i in range(k)) / k


def run_evaluation(
    tasks: list[dict],
    model: str,
    args: argparse.Namespace,
) -> tuple[
    dict[str, list[bool]],
    int,
    Optional[dict[str, list[bool]]],
    int,
    Optional[dict[str, list[float]]],
]:
    """Run evaluation for one model over all tasks.

    Returns ``(results_by_type, no_answer, logprobs_results_or_None, no_answer_logprobs, brier_by_type_or_None)``.
    With ``--choice-only`` or ``--cot-choice``, requests logprobs, fills ``logprobs_results`` (argmax at choice-letter token vs correct),
    and appends per-task multiclass Brier scores (softmax over A–D at that token vs one-hot correct).
    """
    results_by_type: dict[str, list[bool]] = defaultdict(list)
    no_answer = 0
    total = len(tasks)
    use_choice_only = getattr(args, "choice_only", False)
    use_cot_choice = getattr(args, "cot_choice", False)
    prompt_mode = (
        "cot_choice"
        if use_cot_choice
        else ("choice_only" if use_choice_only else "plain")
    )
    use_logprobs = use_choice_only or use_cot_choice
    logprobs_results: Optional[dict[str, list[bool]]] = defaultdict(list) if use_logprobs else None
    no_answer_logprobs = 0
    brier_by_type: Optional[dict[str, list[float]]] = defaultdict(list) if use_logprobs else None

    for i, task in enumerate(tasks):
        if not args.verbose and not args.debug:
            pct = 100 * (i + 1) // total
            bar_len = 24
            filled = min(bar_len * (i + 1) // total, bar_len)
            bar = "=" * filled + (">" if filled < bar_len else "") + " " * max(0, bar_len - filled - 1)
            sys.stderr.write(f"\r  {model}  Task {i + 1}/{total} ({pct:3}%) [{bar}] ")
            sys.stderr.flush()

        if args.verbose or args.debug:
            print(f"[{model}] Task {i + 1}: calling API (timeout {args.timeout}s)...", file=sys.stderr)
            if args.debug:
                print(
                    f"  task_type={task['task_type']} correct_letter={task.get('correct_letter', 'ABCD'[task['correct_index']])} output_mode={prompt_mode}",
                    file=sys.stderr,
                )
            sys.stderr.flush()

        prompt = build_prompt(task, mode=prompt_mode, n_examples=args.n_examples)
        structured_format: Optional[dict] = None
        if use_choice_only:
            structured_format = CHOICE_ONLY_FORMAT
        elif use_cot_choice:
            structured_format = COT_CHOICE_FORMAT
        if "correct_letter" in task:
            correct_idx = ord(task["correct_letter"].upper()) - ord("A")
        else:
            correct_idx = task["correct_index"]

        pred_argmax: Optional[int] = None
        logprobs_by_letter: Optional[dict[str, float]] = None
        choice_token_probs: Optional[list[float]] = None
        try:
            if use_choice_only:
                num_predict = 256
                if getattr(args, "max_tokens", None) is not None and args.max_tokens > 0:
                    num_predict = args.max_tokens
            elif use_cot_choice:
                num_predict = 2048
                if getattr(args, "max_tokens", None) is not None and args.max_tokens > 0:
                    num_predict = args.max_tokens
            else:
                num_predict = None
                if getattr(args, "max_tokens", None) is not None and args.max_tokens > 0:
                    num_predict = args.max_tokens
            response, resp_data = call_ollama(
                prompt,
                model=model,
                base_url=args.base_url,
                timeout=args.timeout,
                think=False if args.no_thinking else None,
                num_predict=num_predict,
                structured_format=structured_format,
                logprobs=use_logprobs,
            )
            if use_logprobs and logprobs_results is not None:
                pred_argmax, logprobs_by_letter = parse_logprobs_by_letter(resp_data, response)
                if pred_argmax is None:
                    no_answer_logprobs += 1
                    logprobs_results[task["task_type"]].append(False)
                else:
                    logprobs_results[task["task_type"]].append(pred_argmax == correct_idx)
                if (
                    brier_by_type is not None
                    and logprobs_by_letter is not None
                    and _has_finite_letter_logprob(logprobs_by_letter)
                    and 0 <= correct_idx < 4
                ):
                    choice_token_probs = letter_logprobs_to_probs(logprobs_by_letter)
                    if choice_token_probs is not None:
                        brier_by_type[task["task_type"]].append(
                            multiclass_brier_score(choice_token_probs, correct_idx)
                        )
        except Exception as e:
            print(f"[{model}] Task {i + 1}: API error: {e}", file=sys.stderr)
            results_by_type[task["task_type"]].append(False)
            if logprobs_results is not None:
                logprobs_results[task["task_type"]].append(False)
            continue

        if args.debug:
            print(f"[{model}] Task {i + 1} raw response: {response!r}", file=sys.stderr)
        if use_logprobs and logprobs_by_letter is not None and (args.verbose or args.debug):
            def _fmt_lp(v: float) -> str:
                if v == float("-inf"):
                    return "-inf"
                return f"{v:.4f}"

            lp_str = " ".join(f"{c}={_fmt_lp(logprobs_by_letter.get(c, float('-inf')))}" for c in "ABCD")
            am_letter = "ABCD"[pred_argmax] if pred_argmax is not None and 0 <= pred_argmax < 4 else "?"
            extra = ""
            sj = parse_structured_choice(response)
            if sj is not None and pred_argmax is not None and sj != pred_argmax:
                extra = f"  JSON choice={'ABCD'[sj]} (choice-token-argmax={am_letter})"
            elif sj is not None:
                extra = f"  JSON choice={'ABCD'[sj]}"
            brier_part = ""
            if choice_token_probs is not None and 0 <= correct_idx < 4:
                br = multiclass_brier_score(choice_token_probs, correct_idx)
                p_str = " ".join(f"{c}={choice_token_probs[j]:.3f}" for j, c in enumerate("ABCD"))
                brier_part = f"  Brier={br:.4f} ({p_str})"
            print(
                f"[{model}] Task {i + 1} logprobs (choice token only): {lp_str} -> argmax={am_letter}{extra}{brier_part}",
                file=sys.stderr,
            )

        pred = parse_answer(
            response,
            answer_options=task["answer_options"],
            correct_index=correct_idx,
        )
        if pred is None:
            no_answer += 1
            correct = False
            if args.verbose or args.debug:
                print(f"[{model}] Task {i + 1} ({task['task_type']}): no parseable answer -> {response[:120]!r}")
        else:
            correct = pred == correct_idx
            if args.verbose or args.debug:
                pred_letter = "ABCD"[pred] if 0 <= pred < 4 else "?"
                correct_letter = "ABCD"[correct_idx] if 0 <= correct_idx < 4 else "?"
                print(f"[{model}] Task {i + 1} ({task['task_type']}): pred={pred_letter} correct={correct_letter} -> {'OK' if correct else 'WRONG'}")

        results_by_type[task["task_type"]].append(correct)

    brier_out: Optional[dict[str, list[float]]] = None
    if brier_by_type is not None:
        brier_out = {k: list(v) for k, v in brier_by_type.items()}
    return results_by_type, no_answer, logprobs_results, no_answer_logprobs, brier_out


def print_results(
    results_by_type: dict[str, list[bool]],
    no_answer: int,
    total: int,
    model_label: str = "",
    logprobs_results: Optional[dict[str, list[bool]]] = None,
    no_answer_logprobs: int = 0,
    brier_by_type: Optional[dict[str, list[float]]] = None,
) -> None:
    """Print summary for one model."""
    total_correct = sum(sum(r) for r in results_by_type.values())
    if model_label:
        print(f"--- Results ({model_label}) ---")
    else:
        print("--- Results ---")
    print(f"Total: {total_correct}/{total} ({100 * total_correct / total:.1f}%)")
    print(f"Unparseable answers: {no_answer}")
    print("By task type:")
    for t in sorted(results_by_type.keys()):
        r = results_by_type[t]
        n = len(r)
        c = sum(r)
        print(f"  {t}: {c}/{n} ({100 * c / n:.1f}%)")
    if logprobs_results is not None:
        lp_total = sum(sum(r) for r in logprobs_results.values())
        print(f"Logprobs (choice-letter token argmax vs correct): {lp_total}/{total} ({100 * lp_total / total:.1f}%)")
        print(f"  No aligned choice token / no A–D mass at that step: {no_answer_logprobs}")
        for t in sorted(logprobs_results.keys()):
            r = logprobs_results[t]
            n = len(r)
            c = sum(r)
            print(f"  {t} (logprobs): {c}/{n} ({100 * c / n:.1f}%)")
    if brier_by_type:
        flat = [x for xs in brier_by_type.values() for x in xs]
        if flat:
            print(
                "Brier score (4-class, softmax p(A–D) at choice-letter token vs one-hot correct): "
                f"mean={statistics.mean(flat):.4f}  (n={len(flat)}/{total}; lower=better, 0=perfect)"
            )
            print(
                "  Formula: (1/4) * sum_{k in A,B,C,D} (p_k - 1[k correct])^2. "
                "Probabilities from softmax over observed logprobs (missing letters floored)."
            )
            for t in sorted(brier_by_type.keys()):
                xs = brier_by_type[t]
                if xs:
                    print(f"  {t} (Brier): mean={statistics.mean(xs):.4f} (n={len(xs)})")
    print()


def print_debug_brier_comparison_table(
    all_results: list[
        tuple[
            str,
            dict[str, list[bool]],
            int,
            Optional[dict[str, list[bool]]],
            int,
            Optional[dict[str, list[float]]],
        ]
    ],
) -> None:
    """After debug runs, print mean Brier (per task type × model) for side-by-side comparison.

    Brier values exist only with ``--choice-only`` (logprobs path). Otherwise prints a short notice.
    """
    task_types: set[str] = set()
    any_brier = False
    for _model, results_by_type, _na, _lr, _nalp, brier_bt in all_results:
        task_types |= set(results_by_type.keys())
        if brier_bt:
            any_brier = True
            task_types |= set(brier_bt.keys())
    cols = sorted(task_types)

    print("\n" + "=" * 72)
    print("Brier comparison (debug): mean score by model × task type  (lower = better)")
    print("=" * 72)
    if not any_brier:
        print(
            "No Brier data: re-run debug with --choice-only or --cot-choice to enable logprobs and per-task Brier."
        )
        print("=" * 72 + "\n")
        return

    w_model = max(12, max(len(m) for m, *_ in all_results) + 1)
    w_cell = max(10, max(len(c) for c in cols) + 2, 12)

    def cell_mean(xs: list[float]) -> str:
        if not xs:
            return "—".rjust(w_cell)
        return f"{statistics.mean(xs):.4f}".rjust(w_cell)

    header = "model".ljust(w_model) + "".join(c.ljust(w_cell) for c in cols) + "overall".rjust(w_cell)
    print(header)
    print("-" * (w_model + w_cell * (len(cols) + 1)))

    column_values: dict[str, list[float]] = {c: [] for c in cols}

    for model, _results_by_type, _na, _lr, _nalp, brier_bt in all_results:
        row_flat: list[float] = []
        parts = [model.ljust(w_model)]
        for c in cols:
            xs = (brier_bt or {}).get(c) or []
            parts.append(cell_mean(xs))
            if xs:
                column_values[c].extend(xs)
                row_flat.extend(xs)
        ov = f"{statistics.mean(row_flat):.4f}".rjust(w_cell) if row_flat else "—".rjust(w_cell)
        parts.append(ov)
        print("".join(parts))

    print("-" * (w_model + w_cell * (len(cols) + 1)))
    footer_label = "col mean".ljust(w_model)
    footer_parts = [footer_label]
    for c in cols:
        xs = column_values[c]
        footer_parts.append(cell_mean(xs))
    all_b = [x for xs in column_values.values() for x in xs]
    footer_ov = f"{statistics.mean(all_b):.4f}".rjust(w_cell) if all_b else "—".rjust(w_cell)
    footer_parts.append(footer_ov)
    print("".join(footer_parts))
    print(
        "Per-cell: mean of per-task Brier scores for that (model, task_type); "
        "counts may be < task count if choice-token logprobs were missing."
    )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 models on ravens-numerical-tasks via Ollama API (HTTP)."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="tasks.json",
        help="Path to tasks JSON (default: tasks.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:30b",
        help="Ollama model name (default: qwen3:30b). Ignored when --debug (all DEBUG_MODELS run).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL: local or remote host (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of tasks to run (default: 10)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=0,
        metavar="N",
        help="In-context demos from IN_CONTEXT_EXAMPLES (0=none, 1=one, 3=all three per task type).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds per task (default: 300)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking for thinking-capable models (Ollama think=false).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max response tokens (Ollama num_predict). Omit for no limit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each task and prediction",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run all DEBUG_MODELS, print raw responses, report results per model.",
    )
    structured_out = parser.add_mutually_exclusive_group()
    structured_out.add_argument(
        "--choice-only",
        action="store_true",
        help="Constrain output with Ollama JSON format (schema: choice A/B/C/D). Sets think=false, JSON prompt, logprobs at choice-letter token, accuracy + mean Brier (softmax p vs one-hot).",
    )
    structured_out.add_argument(
        "--cot-choice",
        action="store_true",
        help="JSON with required string 'reasoning' and enum 'choice' (A–D). Chain-of-thought in reasoning; logprobs/Brier use the choice letter token only (same as --choice-only).",
    )

    args = parser.parse_args()

    path = Path(args.tasks)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data)
    if args.limit:
        tasks = tasks[: args.limit]
    total = len(tasks)

    if args.debug:
        models = DEBUG_MODELS
        all_results: list[
            tuple[
                str,
                dict[str, list[bool]],
                int,
                Optional[dict[str, list[bool]]],
                int,
                Optional[dict[str, list[float]]],
            ]
        ] = []
        try:
            for model in models:
                if not args.verbose:
                    sys.stderr.write("\r" + " " * 60 + "\r")
                    sys.stderr.flush()
                print(f"\n=== Evaluating {model} ===\n", file=sys.stderr)
                results_by_type, no_answer, logprobs_results, no_answer_lp, brier_bt = run_evaluation(
                    tasks, model, args
                )
                all_results.append(
                    (model, results_by_type, no_answer, logprobs_results, no_answer_lp, brier_bt)
                )
        finally:
            if not args.verbose:
                sys.stderr.write("\r" + " " * 60 + "\r")
                sys.stderr.flush()

        print("\n" + "=" * 60)
        if args.cot_choice:
            mode = "CoT JSON (reasoning+choice) + logprobs"
        elif args.choice_only:
            mode = "choice-only JSON + logprobs"
        else:
            mode = "plain letter"
        print(f"SUMMARY: All models (same data, {mode} prompt)")
        print("=" * 60)
        for model, results_by_type, no_answer, logprobs_results, no_answer_lp, brier_bt in all_results:
            print_results(
                results_by_type,
                no_answer,
                total,
                model_label=model,
                logprobs_results=logprobs_results,
                no_answer_logprobs=no_answer_lp,
                brier_by_type=brier_bt,
            )
        print_debug_brier_comparison_table(all_results)
    else:
        try:
            results_by_type, no_answer, logprobs_results, no_answer_lp, brier_bt = run_evaluation(
                tasks, args.model, args
            )
        finally:
            if not args.verbose:
                sys.stderr.write("\r" + " " * 50 + "\r")
                sys.stderr.flush()
        print_results(
            results_by_type,
            no_answer,
            total,
            logprobs_results=logprobs_results,
            no_answer_logprobs=no_answer_lp,
            brier_by_type=brier_bt,
        )


if __name__ == "__main__":
    main()
