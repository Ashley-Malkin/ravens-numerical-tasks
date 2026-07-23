"""Append Modal / vLLM ravens_numerical eval results to experiments.md."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

TASK_TYPE_ORDER = (
    "combine",
    "constancy",
    "constancy_row",
    "intersection",
    "pattern",
    "pattern_tuple",
    "progression",
)


def _pct(accuracy: float) -> str:
    return f"{accuracy * 100:.1f}%"


def _format_task_breakdown(by_task_type: dict[str, Any]) -> str:
    parts: list[str] = []
    seen = set(by_task_type)
    for task_type in TASK_TYPE_ORDER:
        if task_type not in by_task_type:
            continue
        row = by_task_type[task_type]
        parts.append(
            f"{task_type} {_pct(row['accuracy'])} ({row['correct']}/{row['total']})"
        )
        seen.discard(task_type)
    for task_type in sorted(seen):
        row = by_task_type[task_type]
        parts.append(
            f"{task_type} {_pct(row['accuracy'])} ({row['correct']}/{row['total']})"
        )
    return " · ".join(parts)


def _format_model_section(label: str, summary: dict[str, Any]) -> list[str]:
    model_id = summary.get("model_id", label)
    lines = [
        f"### {model_id}",
        f"- **Overall:** {_pct(summary['accuracy'])} "
        f"({summary['correct']}/{summary['total']})",
        f"- **By task:** {_format_task_breakdown(summary['by_task_type'])}",
    ]
    return lines


def format_experiment_entry(
    *,
    run_label: str,
    model_summaries: dict[str, dict[str, Any]],
    max_tasks: Optional[int] = None,
    accuracy_delta: Optional[float] = None,
    settings_note: str = "Modal vLLM (`modal_eval.py`); `--n-examples 0`",
    step: int | None = None,
) -> str:
    """Return a markdown block for one logged experiment run.

    The H2 header includes the UTC run time. When ``step`` is set (checkpoint
    evals), a separate ``(step=N)`` suffix is appended after the tasks note so
    parsers that match ``(max_tasks=N)`` keep working.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tasks_note = "all tasks" if max_tasks is None else f"max_tasks={max_tasks}"
    header = f"## {ts} — {run_label} ({tasks_note})"
    if step is not None:
        header = f"{header} (step={step})"

    lines = [
        header,
        "",
        f"_Logged {ts}. {settings_note}_",
        "",
    ]
    for _key, summary in model_summaries.items():
        lines.extend(_format_model_section(_key, summary))
        lines.append("")

    if accuracy_delta is not None:
        sign = "+" if accuracy_delta >= 0 else ""
        lines.append(
            f"**Δ (Qwen3 − Pythia):** {sign}{accuracy_delta * 100:.1f} pp"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def append_experiment_entry(
    experiments_path: Path,
    *,
    run_label: str,
    model_summaries: dict[str, dict[str, Any]],
    max_tasks: Optional[int] = None,
    accuracy_delta: Optional[float] = None,
    settings_note: str = "Modal vLLM (`modal_eval.py`); `--n-examples 0`",
    file_preamble: str | None = None,
    step: int | None = None,
) -> None:
    """Append one experiment entry to an experiments markdown log."""
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    if not experiments_path.is_file():
        preamble = file_preamble or (
            "# Ravens numerical experiments (Modal / vLLM)\n\n"
            "Auto-appended by `baby_reasoning_eval/modal_eval.py` after each run.\n\n"
        )
        experiments_path.write_text(preamble, encoding="utf-8")
    entry = format_experiment_entry(
        run_label=run_label,
        model_summaries=model_summaries,
        max_tasks=max_tasks,
        accuracy_delta=accuracy_delta,
        settings_note=settings_note,
        step=step,
    )
    with experiments_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Appended experiment log → {experiments_path}", flush=True)
