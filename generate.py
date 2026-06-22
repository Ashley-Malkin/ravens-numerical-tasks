#!/usr/bin/env python3
"""CLI for generating Raven's numerical reasoning tasks."""

import argparse
import json
import random
import sys

from generator import (
    generate_constancy_task,
    generate_combine_task,
    generate_intersection_task,
    generate_pattern_task,
    generate_pattern_tuple_task,
    generate_progression_task,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numerical reasoning tasks inspired by Raven's Standard Progressive Matrices."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of tasks to generate (default: 10)",
    )
    parser.add_argument(
        "--type",
        choices=[
            "constancy",
            "pattern",
            "pattern_tuple",
            "progression",
            "combine",
            "intersection",
            "both",
            "all",
        ],
        default="both",
        help="Task type: constancy, pattern, pattern_tuple, progression, combine, intersection, both, or all (default: both)",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=1,
        help="Minimum number in matrix range (default: 1)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=20,
        help="Maximum number in matrix range (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tasks.json",
        help="Output JSON file path (default: tasks.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.min > args.max:
        print("Error: --min must be less than or equal to --max", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    tasks: list[dict] = []

    type_sequence: list[str] = []
    if args.type == "constancy":
        type_sequence = ["constancy"] * args.count
    elif args.type == "pattern":
        type_sequence = ["pattern"] * args.count
    elif args.type == "pattern_tuple":
        type_sequence = ["pattern_tuple"] * args.count
    elif args.type == "progression":
        type_sequence = ["progression"] * args.count
    elif args.type == "combine":
        type_sequence = ["combine"] * args.count
    elif args.type == "intersection":
        type_sequence = ["intersection"] * args.count
    elif args.type == "both":
        for i in range(args.count):
            type_sequence.append("constancy" if i % 2 == 0 else "pattern")
    else:  # all
        cycle = [
            "constancy",
            "pattern",
            "pattern_tuple",
            "progression",
            "combine",
            "intersection",
        ]
        for i in range(args.count):
            type_sequence.append(cycle[i % len(cycle)])

    for task_type in type_sequence:
        if task_type == "constancy":
            task = generate_constancy_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "pattern":
            task = generate_pattern_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "pattern_tuple":
            task = generate_pattern_tuple_task(rng=rng)
        elif task_type == "progression":
            task = generate_progression_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "combine":
            task = generate_combine_task(rng=rng)
        elif task_type == "intersection":
            task = generate_intersection_task(
                digit_min=args.min, digit_max=args.max, rng=rng
            )
        else:
            raise ValueError(f"unknown task type: {task_type!r}")
        tasks.append(task)

    with open(args.output, "w") as f:
        json.dump({"tasks": tasks}, f, indent=2)

    print(f"Generated {len(tasks)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
