#!/usr/bin/env python3
"""Write nested stratified N-scaling pilots from existing SFT train.jsonl.

Nested: S_50 ⊂ S_100 ⊂ S_250 ⊂ S_500 ⊂ S_1000, stratified across task types.
Does not regenerate train/val or overwrite babylm_finetune/data/sft/pilots/train_n*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ravens_numerical.generation.generator import TASK_TYPE_CYCLE  # noqa: E402

DEFAULT_SEED = 20260730
DEFAULT_SIZES = (50, 100, 250, 500, 1000)
DEFAULT_TRAIN = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "train.jsonl"
DEFAULT_OUT = REPO_ROOT / "babylm_finetune" / "data" / "sft" / "pilots" / "n_scaling"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quotas(n: int, types: list[str]) -> dict[str, int]:
    t = len(types)
    base = n // t
    rem = n % t
    return {tt: base + (1 if i < rem else 0) for i, tt in enumerate(types)}


def nested_stratified_pilots(
    rows: list[dict],
    sizes: tuple[int, ...],
    rng: random.Random,
) -> dict[int, list[dict]]:
    """Build nested stratified subsets keyed by N (ascending sizes)."""
    if not sizes:
        return {}
    if any(s <= 0 for s in sizes):
        raise ValueError("all sizes must be positive")
    max_n = max(sizes)
    if max_n > len(rows):
        raise ValueError(f"max size {max_n} exceeds train rows ({len(rows)})")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["task_type"])].append(row)

    types = [tt for tt in TASK_TYPE_CYCLE if by_type.get(tt)]
    if not types:
        raise ValueError("no task types found in train rows")

    for tt in types:
        rng.shuffle(by_type[tt])
        if len(by_type[tt]) < _quotas(max_n, types)[tt]:
            raise ValueError(
                f"type {tt!r} has {len(by_type[tt])} rows but needs "
                f"{_quotas(max_n, types)[tt]} for N={max_n}"
            )

    out: dict[int, list[dict]] = {}
    for n in sorted(set(sizes)):
        quotas = _quotas(n, types)
        selected: list[dict] = []
        for tt in types:
            selected.extend(by_type[tt][: quotas[tt]])
        # Deterministic order within each N (types already ordered).
        # Final shuffle with a size-specific child RNG derived from the pool order
        # would break nesting; keep type-block order then a shared permutation
        # of indices that is a prefix-compatible shuffle of the max set.
        out[n] = selected
        assert len(selected) == n, (n, len(selected))

    # Verify nesting via fingerprint sets (prompt+completion+task_type).
    def _fp(row: dict) -> str:
        return json.dumps(
            {
                "prompt": row.get("prompt"),
                "completion": row.get("completion"),
                "task_type": row.get("task_type"),
            },
            sort_keys=True,
        )

    ordered = sorted(out)
    for smaller, larger in zip(ordered, ordered[1:]):
        small_set = {_fp(r) for r in out[smaller]}
        large_set = {_fp(r) for r in out[larger]}
        if not small_set <= large_set:
            raise RuntimeError(f"nesting violated: N={smaller} not subset of N={larger}")

    # Shuffle each size with a deterministic RNG forked from size so training
    # order differs by N but membership stays nested. Use index permutation on
    # the max set instead.
    max_rows = out[max_n]
    order = list(range(len(max_rows)))
    rng.shuffle(order)
    # Map fingerprint -> position in shuffled max list for stable subset order.
    fp_to_rank = {_fp(max_rows[i]): rank for rank, i in enumerate(order)}
    for n in out:
        out[n] = sorted(out[n], key=lambda r: fp_to_rank[_fp(r)])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Full SFT train JSONL (default: babylm_finetune/data/sft/train.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for nested pilots",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated nested sizes (default: 50,100,250,500,1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    sizes = tuple(int(p.strip()) for p in args.sizes.split(",") if p.strip())
    if not sizes:
        raise SystemExit("no sizes given")

    train_path = args.train_file.resolve()
    if not train_path.is_file():
        raise SystemExit(f"train file not found: {train_path}")

    rows = _load_jsonl(train_path)
    rng = random.Random(args.seed)
    pilots = nested_stratified_pilots(rows, sizes, rng)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    size_meta: dict[str, dict] = {}
    for n, pilot_rows in sorted(pilots.items()):
        path = out_dir / f"train_n{n}.jsonl"
        _write_jsonl(path, pilot_rows)
        counts = dict(Counter(str(r["task_type"]) for r in pilot_rows))
        size_meta[str(n)] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "count": len(pilot_rows),
            "by_task_type": counts,
        }
        print(f"Wrote N={n} ({len(pilot_rows)}) -> {path}")

    manifest = {
        "seed": args.seed,
        "sizes": list(sizes),
        "train_file": str(train_path.relative_to(REPO_ROOT)),
        "train_count": len(rows),
        "nested": True,
        "pilots": size_meta,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
