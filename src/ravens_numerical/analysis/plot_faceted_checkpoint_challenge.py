#!/usr/bin/env python3
"""7-subtask vs challenge accuracy over training tokens (Pythia / OLMo 2).

Each facet overlays two suites as color families:

- **7-subtask** (green): Easy Ravens ``tasks.json`` overall (7 types)
- **Challenge** (orange): ``challenge_tasks.json`` overall (3 types)

Shades within a family are model size. Checkpoint lines are Pythia 70M / 160M /
1B and OLMo 2 1B / 7B / 13B. Stars mark full models (Pythia ladder at ~300B; OLMo 2 1B / 7B / 13B at
~4T / 3.9T from ``complete.json`` 7-type and 3-type subsets).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ravens_numerical.analysis.dump_scores import (
    default_index,
    overlay_log_scores,
    trial_correct,
)
from ravens_numerical.models.registry import (
    OLMO2_BASE_MODELS,
    OLMO2_CHECKPOINT_SCALING_MODELS,
    PYTHIA_CHECKPOINT_SCALING_MODELS,
    PYTHIA_FULL_TRAINING_TOKENS_B,
    PYTHIA_SCALING_MODELS,
    base_model_id,
    is_olmo2_base_model,
    is_olmo2_checkpoint_model_id,
    is_pythia_checkpoint_model_id,
    is_pythia_model,
    olmo2_tokens_billions_from_revision,
    parse_checkpoint_model_id,
    parse_params_billions,
    pythia_step_from_revision,
    pythia_tokens_billions_at_step,
)
from ravens_numerical.paths import (
    EXPERIMENTS_MD,
    OLMO_CHECKPOINTS_MD,
    PLOTS_DIR,
    PYTHIA_CHECKPOINTS_MD,
    RUNS_DIR,
)

_SUITES = ("easy7", "challenge")
_SUITE_LABELS = {"easy7": "7-subtask", "challenge": "Challenge"}
_CHANCE_PCT = 25.0
_EXCLUDED_OLMO_TOKEN_BUDGETS_B = {51.0}
_MIN_LOG_X_B = 0.0005

_MODEL_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_OVERALL_RE = re.compile(r"\*\*Overall:\*\* ([0-9.]+)%")
_BY_TASK_RE = re.compile(r"\*\*By task:\*\* (.+)$", re.MULTILINE)
_OVERALL_N_RE = re.compile(r"\*\*Overall:\*\* [0-9.]+% \((\d+)/(\d+)\)")

_PYTHIA_LABELS = {
    "EleutherAI/pythia-70m-deduped": "70M",
    "EleutherAI/pythia-160m-deduped": "160M",
    "EleutherAI/pythia-410m-deduped": "410M",
    "EleutherAI/pythia-1b-deduped": "1B",
    "EleutherAI/pythia-1.4b-deduped": "1.4B",
    "EleutherAI/pythia-2.8b-deduped": "2.8B",
    "EleutherAI/pythia-6.9b-deduped": "6.9B",
    "EleutherAI/pythia-12b-deduped": "12B",
}
_OLMO_LABELS = {
    "allenai/OLMo-2-0425-1B": "1B",
    "allenai/OLMo2-7B-1124": "7B",
    "allenai/OLMo-2-13B-1124": "13B",
}
_OLMO_FULL_TOKENS_B = {
    "allenai/OLMo-2-0425-1B": 4000.0,
    "allenai/OLMo2-7B-1124": 3900.0,
    "allenai/OLMo-2-13B-1124": 3900.0,
}
_EASY7_TYPES = frozenset(
    {
        "combine",
        "constancy",
        "constancy_row",
        "intersection",
        "pattern",
        "pattern_tuple",
        "progression",
    }
)
_CHALLENGE_TYPES = frozenset(
    {"distribution_of_three", "progression_plus_n", "tuple_grid"}
)


def _iter_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    yield from (
        part.strip()
        for part in re.split(
            r"(?=^## \d{4}-\d{2}-\d{2} )", text, flags=re.MULTILINE
        )
        if part.strip().startswith("## ")
    )


def _header(block: str) -> str:
    return next((line for line in block.splitlines() if line.strip()), "")


def _overall_n(block: str) -> int | None:
    match = _OVERALL_N_RE.search(block)
    return int(match.group(2)) if match else None


def _block_suite(block: str) -> str | None:
    """Classify a ravens forced-choice block as 7-subtask or challenge."""
    header = _header(block)
    if "forced_choice" not in block:
        return None
    if "tasks_5digit" in block:
        return None
    if ", webb)" in header or "task-type webb" in block:
        return None
    if ", aba)" in header or "task-type aba" in block:
        return None
    if ", ravens)" not in header and "task-type ravens" not in block:
        return None
    n = _overall_n(block)
    if "challenge_tasks" in block:
        return "challenge"
    task_line = _BY_TASK_RE.search(block)
    by_task = task_line.group(1) if task_line else ""
    if "tuple_grid" in by_task:
        return "challenge" if n in (150, None) else None
    if n == 14:
        return None
    if "(max_tasks=140)" in header or n in (140, 350):
        return "easy7"
    return None


def _subset_accuracy(data: list[dict], types: frozenset[str]) -> float | None:
    ok = 0
    n = 0
    for row in data:
        task = row["stimulus"]["metadata"]["task"]["task_type"]
        if task not in types:
            continue
        n += 1
        ok += int(trial_correct(row))
    if n == 0:
        return None
    return 100.0 * ok / n


def olmo_full_from_complete_dumps() -> dict[str, dict[str, float]]:
    """Score Hub OLMo 2 finals from latest 10-type ``complete.json`` dumps.

    Checkpoint challenge evals were saved under the same Hub folder as the
    full model, so a naive challenge-suite lookup picks the wrong dump.
    The n=500 complete dumps are the full models; 7-type / 3-type subsets
    match the two suites on this plot.
    """
    out: dict[str, dict[str, float]] = {suite: {} for suite in _SUITES}
    if not RUNS_DIR.is_dir():
        return out
    for model_id in OLMO2_BASE_MODELS:
        tag = model_id.replace("/", "--")
        latest: tuple[str, list[dict]] | None = None
        for path in (RUNS_DIR / tag).glob(
            "*/ravens_numerical/0_examples_completion.json"
        ):
            stamp = path.parts[-3]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            types = {
                row["stimulus"]["metadata"]["task"]["task_type"] for row in data
            }
            if len(data) != 500 or not _CHALLENGE_TYPES <= types:
                continue
            if latest is None or stamp > latest[0]:
                latest = (stamp, data)
        if latest is None:
            continue
        _stamp, data = latest
        easy = _subset_accuracy(data, _EASY7_TYPES)
        chal = _subset_accuracy(data, _CHALLENGE_TYPES)
        if easy is not None:
            out["easy7"][model_id] = easy
        if chal is not None:
            out["challenge"][model_id] = chal
    return out


def parse_latest(*paths: Path) -> dict[str, dict[str, float]]:
    """Return ``suite -> model_id -> latest overall accuracy percent``."""
    latest: dict[str, dict[str, float]] = {suite: {} for suite in _SUITES}
    rank: dict[tuple[str, str], int] = {}
    for path in paths:
        if not path.is_file():
            continue
        for block in _iter_blocks(path):
            suite = _block_suite(block)
            if suite is None:
                continue
            model_match = _MODEL_RE.search(block)
            overall_match = _OVERALL_RE.search(block)
            if not model_match or not overall_match:
                continue
            model_id = model_match.group(1).strip()
            n = _overall_n(block) or 0
            quality = 2 if (suite == "easy7" and n == 350) else 1
            key = (suite, model_id)
            if rank.get(key, 0) > quality:
                continue
            overall = float(overall_match.group(1))
            if "@" not in model_id:
                dump_suite = "challenge" if suite == "challenge" else "regular"
                overall, _ = overlay_log_scores(
                    model_id, dump_suite, overall, {}
                )
            latest[suite][model_id] = float(overall)
            rank[key] = quality

    # Overlay Hub Pythia dumps. OLMo Hub challenge dumps mix checkpoints with
    # the full-model folder; score finals from complete.json instead.
    for model_id in PYTHIA_SCALING_MODELS:
        for suite, dump_suite in (("easy7", "regular"), ("challenge", "challenge")):
            hit = default_index().lookup(model_id, dump_suite)
            if hit is None:
                continue
            latest[suite][model_id] = hit.overall
    for suite, scores in olmo_full_from_complete_dumps().items():
        latest[suite].update(scores)
    return latest


def _checkpoint_tokens_b(model_id: str) -> float | None:
    _, revision = parse_checkpoint_model_id(model_id)
    if revision is None:
        return None
    if is_pythia_checkpoint_model_id(model_id):
        step = pythia_step_from_revision(revision)
        return pythia_tokens_billions_at_step(step) if step is not None else None
    if is_olmo2_checkpoint_model_id(model_id):
        tokens_b = olmo2_tokens_billions_from_revision(revision)
        if (
            tokens_b is None
            or tokens_b <= 0
            or tokens_b in _EXCLUDED_OLMO_TOKEN_BUDGETS_B
        ):
            return None
        return tokens_b
    return None


def checkpoint_series(
    latest: dict[str, dict[str, float]],
    *,
    family: str,
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    bases = (
        PYTHIA_CHECKPOINT_SCALING_MODELS
        if family == "pythia"
        else OLMO2_CHECKPOINT_SCALING_MODELS
    )
    out = {suite: {base: [] for base in bases} for suite in _SUITES}
    for suite, results in latest.items():
        for model_id, accuracy in results.items():
            is_checkpoint = (
                is_pythia_checkpoint_model_id(model_id)
                if family == "pythia"
                else is_olmo2_checkpoint_model_id(model_id)
            )
            if not is_checkpoint:
                continue
            base = base_model_id(model_id)
            if base not in out[suite]:
                continue
            tokens_b = _checkpoint_tokens_b(model_id)
            if tokens_b is not None:
                out[suite][base].append((tokens_b, accuracy))
    for by_model in out.values():
        for points in by_model.values():
            points.sort()
    return out


def full_models(
    latest: dict[str, dict[str, float]],
    *,
    family: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {suite: {} for suite in _SUITES}
    for suite, results in latest.items():
        for model_id, accuracy in results.items():
            if "@" in model_id:
                continue
            if family == "pythia" and is_pythia_model(model_id):
                out[suite][model_id] = accuracy
            elif family == "olmo" and is_olmo2_base_model(model_id):
                out[suite][model_id] = accuracy
    return out


def _shades(cmap, count: int) -> list:
    if count == 1:
        return [cmap(0.75)]
    return [cmap(0.88 - index * 0.45 / (count - 1)) for index in range(count)]


def _size_label(model_id: str, labels: dict[str, str]) -> str:
    if model_id in labels:
        return labels[model_id]
    params = parse_params_billions(model_id)
    if params is None:
        return model_id.rsplit("/", 1)[-1]
    if params < 1:
        return f"{int(round(params * 1000))}M"
    if params == int(params):
        return f"{int(params)}B"
    return f"{params:g}B"


def plot(
    latest: dict[str, dict[str, float]],
    *,
    output: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise SystemExit("matplotlib required: pip install matplotlib") from exc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), sharex=True, sharey=True)
    suite_cmaps = {
        "easy7": plt.get_cmap("Greens"),
        "challenge": plt.get_cmap("Oranges"),
    }
    suite_style = {"easy7": ("-", "o"), "challenge": ("--", "^")}

    panels = (
        (
            axes[0],
            "Pythia",
            PYTHIA_CHECKPOINT_SCALING_MODELS,
            PYTHIA_SCALING_MODELS,
            _PYTHIA_LABELS,
            checkpoint_series(latest, family="pythia"),
            full_models(latest, family="pythia"),
            {base: PYTHIA_FULL_TRAINING_TOKENS_B for base in PYTHIA_SCALING_MODELS},
        ),
        (
            axes[1],
            "OLMo 2",
            OLMO2_CHECKPOINT_SCALING_MODELS,
            OLMO2_CHECKPOINT_SCALING_MODELS,
            _OLMO_LABELS,
            checkpoint_series(latest, family="olmo"),
            full_models(latest, family="olmo"),
            _OLMO_FULL_TOKENS_B,
        ),
    )

    all_x: list[float] = []
    legend_handles: list = []

    for (
        ax,
        family_label,
        line_bases,
        star_pool,
        labels,
        series,
        finals,
        full_tokens,
    ) in panels:
        for suite in _SUITES:
            star_bases = tuple(mid for mid in star_pool if mid in finals[suite])
            color_bases = tuple(dict.fromkeys((*line_bases, *star_bases)))
            colors = dict(
                zip(color_bases, _shades(suite_cmaps[suite], max(len(color_bases), 1)))
            )
            linestyle, marker = suite_style[suite]
            for base in line_bases:
                points = series[suite].get(base) or []
                color = colors[base]
                if points:
                    x, y = zip(*points)
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2,
                        markersize=6,
                    )
                    all_x.extend(x)
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2,
                        label=(
                            f"{family_label} {_size_label(base, labels)} — "
                            f"{_SUITE_LABELS[suite]}"
                        ),
                    )
                )
            extra_stars = [b for b in star_bases if b not in line_bases]
            for base in star_bases:
                if base not in finals[suite]:
                    continue
                ax.plot(
                    full_tokens[base],
                    finals[suite][base],
                    marker="*",
                    color=colors[base],
                    markersize=14 if base in line_bases else 16,
                    linestyle="none",
                    zorder=4,
                )
                all_x.append(full_tokens[base])
            for base in extra_stars:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors[base],
                        marker="*",
                        linestyle="none",
                        markersize=12,
                        label=(
                            f"{family_label} {_size_label(base, labels)} "
                            f"(full) — {_SUITE_LABELS[suite]}"
                        ),
                    )
                )

        ax.set_title(family_label)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens seen (billions, log scale)")
        ax.axhline(
            _CHANCE_PCT, color="gray", linestyle="--", linewidth=1, alpha=0.7
        )
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Overall accuracy (%)")
    if all_x:
        axes[0].set_xlim(max(_MIN_LOG_X_B, min(all_x) * 0.75), max(all_x) * 1.15)

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                linestyle="none",
                markersize=12,
                label="Full model",
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"Chance ({_CHANCE_PCT:g}%)",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.36, wspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_MD)
    parser.add_argument(
        "--pythia-checkpoints", type=Path, default=PYTHIA_CHECKPOINTS_MD
    )
    parser.add_argument("--olmo-checkpoints", type=Path, default=OLMO_CHECKPOINTS_MD)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    latest = parse_latest(
        args.experiments, args.pythia_checkpoints, args.olmo_checkpoints
    )
    for suite in _SUITES:
        n_ck = sum(
            1
            for mid in latest[suite]
            if is_pythia_checkpoint_model_id(mid)
            or is_olmo2_checkpoint_model_id(mid)
        )
        n_full = sum(1 for mid in latest[suite] if "@" not in mid)
        print(
            f"{_SUITE_LABELS[suite]}: {n_ck} checkpoints, {n_full} full-model ids"
        )
    output = args.output or (
        PLOTS_DIR / "faceted_checkpoints_challenge_forced_choice_scaling.png"
    )
    title = args.title or (
        "7-subtask vs challenge accuracy vs tokens seen — forced choice"
    )
    plot(latest, output=output, title=title)


if __name__ == "__main__":
    main()
