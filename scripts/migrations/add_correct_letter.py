#!/usr/bin/env python3
"""Add correct_letter (A/B/C/D) to each task in tasks.json from correct_index."""

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(__file__).parent / "tasks.json"
    with open(path) as f:
        data = json.load(f)
    tasks = data.get("tasks", data)
    letters = "ABCD"
    for task in tasks:
        idx = task["correct_index"]
        task["correct_letter"] = letters[idx] if 0 <= idx < 4 else "?"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Added correct_letter to {len(tasks)} tasks in {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
