#!/usr/bin/env python3
"""CLI for generating Raven's numerical reasoning tasks."""

import argparse
import json
import random
import sys

from ravens_numerical.generation.generator import (
    LETTERS,
    TASK_TYPE_CYCLE,
    generate_constancy_row_task,
    generate_constancy_task,
    generate_combine_task,
    generate_distribution_of_three_task,
    generate_intersection_task,
    generate_pattern_task,
    generate_pattern_tuple_task,
    generate_progression_plus_n_task,
    generate_progression_task,
    generate_tuple_grid_task,
    PROGRESSION_STEPS,
    _expand_task_to_3x3,
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
            "constancy_row",
            "pattern",
            "pattern_tuple",
            "progression",
            "combine",
            "intersection",
            "distribution_of_three",
            "progression_plus_n",
            "tuple_grid",
            "both",
            "all",
        ],
        default="both",
        help="Task type, both (constancy+pattern), or all (default: both)",
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
    elif args.type == "constancy_row":
        type_sequence = ["constancy_row"] * args.count
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
    elif args.type == "distribution_of_three":
        type_sequence = ["distribution_of_three"] * args.count
    elif args.type == "progression_plus_n":
        type_sequence = ["progression_plus_n"] * args.count
    elif args.type == "tuple_grid":
        type_sequence = ["tuple_grid"] * args.count
    elif args.type == "both":
        for i in range(args.count):
            type_sequence.append("constancy" if i % 2 == 0 else "pattern")
    else:  # all
        for i in range(args.count):
            type_sequence.append(TASK_TYPE_CYCLE[i % len(TASK_TYPE_CYCLE)])

    for task_type in type_sequence:
        if task_type == "constancy":
            task = generate_constancy_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
            task = _expand_task_to_3x3(task)
        elif task_type == "constancy_row":
            task = generate_constancy_row_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "pattern":
            task = generate_pattern_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
            task = _expand_task_to_3x3(task)
        elif task_type == "pattern_tuple":
            task = generate_pattern_tuple_task(rng=rng)
            task = _expand_task_to_3x3(task)
        elif task_type == "progression":
            task = generate_progression_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
            task = _expand_task_to_3x3(task)
        elif task_type == "combine":
            task = generate_combine_task(rng=rng)
        elif task_type == "intersection":
            task = generate_intersection_task(
                digit_min=args.min, digit_max=args.max, rng=rng
            )
        elif task_type == "distribution_of_three":
            task = generate_distribution_of_three_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "progression_plus_n":
            step = PROGRESSION_STEPS[len(tasks) % len(PROGRESSION_STEPS)]
            task = generate_progression_plus_n_task(
                step=step, min_val=args.min, max_val=args.max, rng=rng
            )
        elif task_type == "tuple_grid":
            task = generate_tuple_grid_task(
                min_val=args.min, max_val=args.max, rng=rng
            )
        else:
            raise ValueError(f"unknown task type: {task_type!r}")
        task["correct_letter"] = LETTERS[task["correct_index"]]
        tasks.append(task)

    with open(args.output, "w") as f:
        json.dump({"tasks": tasks}, f, indent=2)

    print(f"Generated {len(tasks)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
