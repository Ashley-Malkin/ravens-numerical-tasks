"""Core task generation logic for Raven's numerical reasoning tasks."""

import random
from typing import Optional

# Type alias: tuple of digits represented as list for JSON serialization
DigitTuple = list[int]


def generate_distractors(
    correct: int,
    min_val: int,
    max_val: int,
    exclude: Optional[frozenset[int]] = None,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """Generate 3 unique distractor options using nearby-number strategy.

    Uses values from {correct ± 1, correct ± 2, ...} within [min_val, max_val].
    Expands range (e.g., ±3, ±4) if fewer than 3 candidates available.
    """
    exclude = exclude or frozenset()
    rng = rng or random.Random()

    candidates = set()
    delta = 1
    while len(candidates) < 3 and delta <= max(max_val - min_val, 10):
        for d in [-delta, delta]:
            v = correct + d
            if min_val <= v <= max_val and v != correct and v not in exclude:
                candidates.add(v)
        delta += 1

    if len(candidates) < 3:
        # Fallback: use any values in range that aren't correct
        for v in range(min_val, max_val + 1):
            if v != correct and v not in exclude and v not in candidates:
                candidates.add(v)
                if len(candidates) >= 3:
                    break

    return rng.sample(list(candidates), min(3, len(candidates)))


def generate_constancy_task(
    min_val: int = 1,
    max_val: int = 20,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a numerical constancy matching task.

    Matrix: all positions same number N except (1,1) which is blank.
    Correct answer: N
    """
    rng = rng or random.Random()
    n = rng.randint(min_val, max_val)

    distractors = generate_distractors(n, min_val, max_val, rng=rng)
    options = [n] + distractors
    rng.shuffle(options)
    correct_index = options.index(n)

    return {
        "task_type": "constancy",
        "matrix": [[n, n], [n, None]],
        "answer_options": options,
        "correct_index": correct_index,
    }


def _generate_constancy_row_distractors(
    correct: int,
    other_rows: tuple[int, int],
    min_val: int,
    max_val: int,
    rng: random.Random,
) -> list[int]:
    """One distractor from another row's value plus two nearby numbers."""
    row_distractor = rng.choice(list(other_rows))
    exclude = frozenset({correct, row_distractor})
    nearby = generate_distractors(correct, min_val, max_val, exclude=exclude, rng=rng)
    distractors = [row_distractor]
    for v in nearby:
        if v not in distractors:
            distractors.append(v)
        if len(distractors) >= 3:
            break
    if len(distractors) < 3:
        for v in range(min_val, max_val + 1):
            if v not in distractors and v != correct:
                distractors.append(v)
            if len(distractors) >= 3:
                break
    return distractors[:3]


def generate_constancy_row_task(
    min_val: int = 1,
    max_val: int = 20,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a row-constancy task (3×3).

    Each row holds a constant value; blank at (2, 2). Correct answer: row 2's value.
    Distractors: one value from another row, plus two nearby numbers.
    """
    rng = rng or random.Random()
    if max_val - min_val + 1 < 3:
        raise ValueError("constancy_row needs at least 3 distinct values in range")

    a, b, c = rng.sample(range(min_val, max_val + 1), 3)
    correct = c
    distractors = _generate_constancy_row_distractors(
        correct, (a, b), min_val, max_val, rng
    )
    options = [correct] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct)

    return {
        "task_type": "constancy_row",
        "matrix": [[a, a, a], [b, b, b], [c, c, None]],
        "answer_options": options,
        "correct_index": correct_index,
    }


def generate_pattern_task(
    min_val: int = 1,
    max_val: int = 20,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a pattern following task.

    Matrix: (0,0)=X, (1,0)=X, (0,1)=Y (Y≠X), (1,1)=blank.
    Correct answer: Y (value at (0,1) - pattern: column stays constant)
    """
    rng = rng or random.Random()

    x = rng.randint(min_val, max_val)
    y = rng.randint(min_val, max_val)
    while y == x:
        y = rng.randint(min_val, max_val)

    distractors = generate_distractors(
        y, min_val, max_val, exclude=frozenset(), rng=rng
    )
    options = [y] + distractors
    rng.shuffle(options)
    correct_index = options.index(y)

    return {
        "task_type": "pattern",
        "matrix": [[x, y], [x, None]],
        "answer_options": options,
        "correct_index": correct_index,
    }


def generate_progression_task(
    min_val: int = 1,
    max_val: int = 20,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a progression task (second column = first column + 1).

    Matrix: (0,0)=A, (0,1)=A+1, (1,0)=B, (1,1)=blank.
    Pattern: each row, second column is one higher than first column.
    Correct answer: B + 1
    """
    rng = rng or random.Random()

    # B+1 must be in range, so B in [min_val, max_val-1]
    b = rng.randint(min_val, max_val - 1) if min_val < max_val else min_val
    correct = b + 1

    distractors = generate_distractors(correct, min_val, max_val, rng=rng)
    options = [correct] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct)

    # A, A+1 for row 0; A must be in [min_val, max_val-1]
    a = rng.randint(min_val, max_val - 1) if min_val < max_val else min_val

    return {
        "task_type": "progression",
        "matrix": [[a, a + 1], [b, None]],
        "answer_options": options,
        "correct_index": correct_index,
    }


def _tuple_to_key(t: DigitTuple) -> tuple:
    """Convert list to hashable tuple for set membership."""
    return tuple(t)


def _generate_tuple_distractors(
    correct: DigitTuple,
    other: DigitTuple,
    digit_min: int = 0,
    digit_max: int = 9,
    rng: Optional[random.Random] = None,
) -> list[DigitTuple]:
    """Generate 3 distractors for tuple pattern task.

    Varies by both digit values and number of digits.
    At least one same-length (different digits) and one different-length.
    """
    rng = rng or random.Random()
    exclude = {_tuple_to_key(correct), _tuple_to_key(other)}
    n = len(correct)

    same_length: list[DigitTuple] = []
    diff_length: list[DigitTuple] = []

    # Same length, different digit values
    if n == 1:
        for d in range(digit_min, digit_max + 1):
            if d != correct[0]:
                cand = [d]
                if _tuple_to_key(cand) not in exclude:
                    same_length.append(cand)
    else:
        for d1 in range(max(1, digit_min), digit_max + 1):
            for d2 in range(digit_min, digit_max + 1):
                cand = [d1, d2]
                if cand != correct and _tuple_to_key(cand) not in exclude:
                    same_length.append(cand)
                if len(same_length) >= 6:
                    break
            if len(same_length) >= 6:
                break

    # Different length
    other_len = 2 if n == 1 else 1
    if other_len == 1:
        for d in range(max(1, digit_min), digit_max + 1):
            cand = [d]
            if _tuple_to_key(cand) not in exclude:
                diff_length.append(cand)
    else:
        for d1 in range(max(1, digit_min), digit_max + 1):
            for d2 in range(digit_min, digit_max + 1):
                cand = [d1, d2]
                if _tuple_to_key(cand) not in exclude:
                    diff_length.append(cand)
                if len(diff_length) >= 6:
                    break
            if len(diff_length) >= 6:
                break

    distractors: list[DigitTuple] = []
    same_available = [c for c in same_length if _tuple_to_key(c) not in exclude]
    diff_available = [c for c in diff_length if _tuple_to_key(c) not in exclude]
    if same_available:
        pick = rng.choice(same_available)
        distractors.append(pick)
        exclude.add(_tuple_to_key(pick))
    if diff_available:
        diff_filtered = [c for c in diff_available if _tuple_to_key(c) not in exclude]
        if diff_filtered:
            pick = rng.choice(diff_filtered)
            distractors.append(pick)
            exclude.add(_tuple_to_key(pick))
    while len(distractors) < 3:
        pool = [c for c in same_length + diff_length if _tuple_to_key(c) not in exclude]
        if not pool:
            break
        pick = rng.choice(pool)
        distractors.append(pick)
        exclude.add(_tuple_to_key(pick))

    return distractors[:3]


def generate_pattern_tuple_task(
    digit_min: int = 0,
    digit_max: int = 9,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a tuple pattern task.

    Matrix: (0,0)=A, (0,1)=B, (1,0)=B, (1,1)=blank.
    Pattern: diagonal - (0,0) matches (1,1), so correct answer is A.
    A and B differ in digit values and/or number of digits.
    Distractors vary by both digit values and number of digits.
    """
    rng = rng or random.Random()

    # A: 1-digit, B: 2-digit (or vice versa) to test both variables
    if rng.random() < 0.5:
        a = [rng.randint(max(1, digit_min), digit_max)]
        d1 = rng.randint(max(1, digit_min), digit_max)
        d2 = rng.randint(digit_min, digit_max)
        b = [d1, d2]
        while _tuple_to_key(b) == _tuple_to_key(a) or b == a:
            d1 = rng.randint(max(1, digit_min), digit_max)
            d2 = rng.randint(digit_min, digit_max)
            b = [d1, d2]
    else:
        d1 = rng.randint(max(1, digit_min), digit_max)
        d2 = rng.randint(digit_min, digit_max)
        b = [d1, d2]
        a = [rng.randint(max(1, digit_min), digit_max)]
        while _tuple_to_key(a) == _tuple_to_key(b) or a == b:
            a = [rng.randint(max(1, digit_min), digit_max)]

    distractors = _generate_tuple_distractors(a, b, digit_min, digit_max, rng=rng)
    options = [list(a)] + [list(d) for d in distractors]
    rng.shuffle(options)
    correct_index = next(i for i, o in enumerate(options) if o == a)

    return {
        "task_type": "pattern_tuple",
        "matrix": [[a, b], [b, None]],
        "answer_options": options,
        "correct_index": correct_index,
    }


def _generate_logic_distractors(
    correct: DigitTuple,
    exclude: set[tuple],
    digit_min: int,
    digit_max: int,
    rng: random.Random,
    single_digit_only: bool = False,
) -> list[DigitTuple]:
    """Generate 3 distractors for logic matrices, excluding given tuple keys."""
    exclude = set(exclude)  # copy to avoid mutating caller
    distractors: list[DigitTuple] = []

    # Collect 1-tuples and 2-tuples as candidates
    one_tuples = [
        [d] for d in range(max(1, digit_min), digit_max + 1)
        if _tuple_to_key([d]) not in exclude
    ]
    two_tuples = [] if single_digit_only else [
        [d1, d2]
        for d1 in range(max(1, digit_min), digit_max + 1)
        for d2 in range(digit_min, digit_max + 1)
        if _tuple_to_key([d1, d2]) not in exclude
    ]

    pool = one_tuples + two_tuples
    while len(distractors) < 3 and pool:
        cand = rng.choice(pool)
        if _tuple_to_key(cand) not in exclude:
            distractors.append(list(cand))
            exclude.add(_tuple_to_key(cand))
        pool = [p for p in pool if _tuple_to_key(p) not in exclude]

    return distractors[:3]


def generate_combine_task(
    digit_min: int = 1,
    digit_max: int = 9,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate a combine task (3x3).

    Each row: col3 = tuple of (col1, col2) - both elements combined.
    Row 2 col3 is blank; correct = [a3, b3].
    """
    rng = rng or random.Random()

    def pick_row() -> tuple[int, int]:
        a = rng.randint(digit_min, digit_max)
        b = rng.randint(digit_min, digit_max)
        while b == a:
            b = rng.randint(digit_min, digit_max)
        return a, b

    (a1, b1), (a2, b2), (a3, b3) = pick_row(), pick_row(), pick_row()

    correct = [a3, b3]
    exclude = {
        (a3, b3),
        (b3, a3),
        (a3,),
        (b3,),
    }
    distractors = _generate_logic_distractors(
        correct, exclude, digit_min, digit_max, rng
    )
    options = [list(correct)] + [list(d) for d in distractors]
    rng.shuffle(options)
    correct_index = next(i for i, o in enumerate(options) if o == correct)

    matrix = [
        [[a1], [b1], [a1, b1]],
        [[a2], [b2], [a2, b2]],
        [[a3], [b3], None],
    ]
    return {
        "task_type": "combine",
        "matrix": matrix,
        "answer_options": options,
        "correct_index": correct_index,
    }


def _one_intersection_row(
    rng: random.Random,
    lo: int,
    hi: int,
) -> tuple[list[int], list[int], int]:
    """Two length-2 lists sharing exactly one element ``c``; return (left, right, c)."""
    for widen in range(8):
        a, b = lo, hi
        if b - a + 1 < 3:
            a, b = min(a, 0), max(b, 9 + widen)
        vals = list(range(a, b + 1))
        if len(vals) < 3:
            continue
        for _ in range(80):
            c = rng.choice(vals)
            others = [v for v in vals if v != c]
            rng.shuffle(others)
            if len(others) < 2:
                continue
            x, y = others[0], others[1]
            left = [c, x]
            right = [c, y]
            rng.shuffle(left)
            rng.shuffle(right)
            if set(left) & set(right) == {c}:
                return left, right, c
    raise RuntimeError("Could not sample intersection row; widen --min/--max")


def generate_intersection_task(
    digit_min: int = 0,
    digit_max: int = 9,
    rng: Optional[random.Random] = None,
) -> dict:
    """Generate an intersection task (3x3).

    Each row: two pairs (length-2 lists) that share **exactly one** element; column 3 is
    that element as a one-element list ``[c]``. The blank is row 2 col 3; the answer is the
    unique shared element of the two pairs in that row.

    ``answer_options`` are four **integers**: the correct value plus three distractors drawn
    from ``[digit_min, digit_max]`` via ``generate_distractors`` (same strategy as other
    scalar tasks).
    """
    rng = rng or random.Random()

    rows_data: list[tuple[list[int], list[int], int]] = []
    matrix: list[list] = []
    for _row in range(3):
        left, right, c = _one_intersection_row(rng, digit_min, digit_max)
        rows_data.append((left, right, c))
        matrix.append([left, right, [c]])

    matrix[2][2] = None
    correct = rows_data[2][2]

    distractors = generate_distractors(correct, digit_min, digit_max, rng=rng)
    if len(distractors) < 3:
        raise RuntimeError(
            "Could not sample 3 distractors for intersection; widen --min/--max"
        )
    options = [correct] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct)

    return {
        "task_type": "intersection",
        "matrix": matrix,
        "answer_options": options,
        "correct_index": correct_index,
    }


def generate_distribution_of_three_task(
    min_val: int = 1,
    max_val: int = 20,
    rng: Optional[random.Random] = None,
) -> dict:
    """Cyclic ordering of three distinct numbers.

    Example::

        [[6, 2, 4],
         [2, 4, 6],
         [4, 6, ?]]   → 2

    Two distractors are the other members of the triple; one is an outsider.
    """
    rng = rng or random.Random()
    a, b, c = rng.sample(range(min_val, max_val + 1), 3)
    row0 = [a, b, c]
    row1 = [b, c, a]
    row2 = [c, a, b]
    correct = row2[2]
    others = [x for x in (a, b, c) if x != correct]
    outsider_pool = [
        v for v in range(min_val, max_val + 1) if v not in (a, b, c)
    ]
    if not outsider_pool:
        outsider = max_val + 1
    else:
        outsider = rng.choice(outsider_pool)
    options = [correct, others[0], others[1], outsider]
    rng.shuffle(options)
    return {
        "task_type": "distribution_of_three",
        "matrix": [row0, row1, [row2[0], row2[1], None]],
        "answer_options": options,
        "correct_index": options.index(correct),
    }


def generate_progression_plus_n_task(
    step: int,
    min_val: int = 1,
    max_val: int = 50,
    rng: Optional[random.Random] = None,
) -> dict:
    """Row-wise arithmetic progression with common difference ``step``.

    Example (``step=2``)::

        [[3, 5, 7],
         [2, 4, 6],
         [9, 11, ?]]  → 13
    """
    rng = rng or random.Random()
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    start_hi = max_val - 2 * step
    if start_hi < min_val:
        raise ValueError(
            f"range [{min_val}, {max_val}] too small for step={step}"
        )

    starts = [rng.randint(min_val, start_hi) for _ in range(3)]
    matrix = [[s, s + step, s + 2 * step] for s in starts]
    correct = matrix[2][2]
    matrix[2][2] = None

    visible = [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
    ]
    exclude = frozenset(visible) | {correct}
    distractors = generate_distractors(
        correct, min_val, max_val, exclude=exclude, rng=rng
    )
    preferred: list[int] = []
    for cand in (
        matrix[2][1],
        matrix[1][2],
        correct - 1 if correct - 1 >= min_val else correct + 1,
        correct + step,
        starts[2] + 1,
    ):
        if (
            min_val <= cand <= max_val
            and cand != correct
            and cand not in preferred
        ):
            preferred.append(cand)
        if len(preferred) >= 3:
            break
    for d in distractors:
        if len(preferred) >= 3:
            break
        if d != correct and d not in preferred:
            preferred.append(d)
    options = [correct] + preferred[:3]
    rng.shuffle(options)
    return {
        "task_type": "progression_plus_n",
        "step": step,
        "matrix": matrix,
        "answer_options": options,
        "correct_index": options.index(correct),
    }


def generate_tuple_grid_task(
    min_val: int = 1,
    max_val: int = 12,
    rng: Optional[random.Random] = None,
) -> dict:
    """Row-constant first coord, column-constant second; order-variant options.

    Example::

        [[(1,3), (1,5), (1,2)],
         [(4,3), (4,5), (4,2)],
         [(8,3), (8,5), ?]]  → (8, 2)
    """
    rng = rng or random.Random()
    keys = rng.sample(range(min_val, max_val + 1), 6)
    row_keys = keys[:3]
    col_keys = keys[3:]
    matrix = [
        [[row_keys[i], col_keys[j]] for j in range(3)] for i in range(3)
    ]
    correct = [row_keys[2], col_keys[2]]
    matrix[2][2] = None

    scalar = row_keys[2]
    swapped = [col_keys[2], row_keys[2]]
    mixed = [row_keys[1], row_keys[2]]
    if mixed == correct or mixed == swapped:
        mixed = [col_keys[0], row_keys[2]]
    if mixed == correct or mixed == swapped:
        mixed = [row_keys[0], col_keys[1]]

    options: list = [scalar, correct, mixed, swapped]

    def _opt_key(o) -> tuple:
        if isinstance(o, list):
            return ("list", tuple(o))
        return ("int", o)

    if len({_opt_key(o) for o in options}) < 4:
        alt = [row_keys[0], col_keys[0]]
        if _opt_key(alt) in {_opt_key(correct), _opt_key(swapped)}:
            alt = [row_keys[1], col_keys[1]]
        options = [scalar, correct, alt, swapped]

    rng.shuffle(options)
    correct_index = next(
        i for i, o in enumerate(options) if _opt_key(o) == _opt_key(correct)
    )
    return {
        "task_type": "tuple_grid",
        "matrix": matrix,
        "answer_options": options,
        "correct_index": correct_index,
        "perm_invariant": False,
    }


LETTERS = "ABCD"

PROGRESSION_STEPS = (2, 3, 5, 7, 10)
CHALLENGE_TYPE_CYCLE: tuple[str, ...] = (
    "distribution_of_three",
    "progression_plus_n",
    "tuple_grid",
)

# Original 7-type round-robin used by ``data/tasks.json``.
LEGACY_TASK_TYPE_CYCLE: tuple[str, ...] = (
    "constancy",
    "constancy_row",
    "pattern",
    "pattern_tuple",
    "progression",
    "combine",
    "intersection",
)

# Round-robin order for ``complete.json`` and ``generate.py --type all``.
TASK_TYPE_CYCLE: tuple[str, ...] = LEGACY_TASK_TYPE_CYCLE + CHALLENGE_TYPE_CYCLE

# Alias for ICL example generation (same order as ``TASK_TYPE_CYCLE``).
_ICL_TASK_TYPES = TASK_TYPE_CYCLE


def interleave_tasks_by_type(
    tasks: list[dict],
    *,
    type_cycle: tuple[str, ...] = TASK_TYPE_CYCLE,
) -> list[dict]:
    """Reorder tasks round-robin by ``task_type`` so ``max_tasks`` slices include every type."""
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_type[str(task["task_type"])].append(task)

    rounds = max((len(by_type[tt]) for tt in type_cycle if by_type.get(tt)), default=0)
    interleaved: list[dict] = []
    for _round in range(rounds):
        for tt in type_cycle:
            bucket = by_type.get(tt, [])
            if _round < len(bucket):
                interleaved.append(bucket[_round])
    return interleaved


def _expand_task_to_3x3(task: dict) -> dict:
    """Expand a generator task to the 3×3 ``tasks.json`` matrix layout."""
    tt = task["task_type"]
    matrix = task["matrix"]

    if tt == "constancy":
        n = matrix[0][0]
        expanded = [[n, n, n], [n, n, n], [n, n, None]]
    elif tt == "pattern":
        x, y = matrix[0][0], matrix[0][1]
        expanded = [[x, y, y], [x, y, y], [x, y, None]]
    elif tt == "pattern_tuple":
        a, b = matrix[0][0], matrix[0][1]
        expanded = [[a, b, b], [b, a, b], [b, b, None]]
    elif tt in (
        "combine",
        "intersection",
        "constancy_row",
        "distribution_of_three",
        "progression_plus_n",
        "tuple_grid",
    ):
        expanded = matrix
    else:
        raise ValueError(f"cannot expand task type: {tt!r}")

    return {**task, "matrix": expanded}


def _generate_icl_candidate(
    task_type: str,
    rng: random.Random,
    *,
    min_val: int = 1,
    max_val: int = 20,
    digit_min: int = 0,
    digit_max: int = 9,
) -> dict:
    """Generate one 3×3 demonstration task of ``task_type``.

    ``min_val`` / ``max_val`` apply to magnitude types (constancy, constancy_row,
    pattern, progression). ``digit_min`` / ``digit_max`` apply to digit/tuple
    types (combine, intersection, pattern_tuple).
    """
    if task_type == "constancy":
        task = generate_constancy_task(min_val=min_val, max_val=max_val, rng=rng)
    elif task_type == "constancy_row":
        return generate_constancy_row_task(
            min_val=min_val, max_val=max_val, rng=rng
        )
    elif task_type == "pattern":
        task = generate_pattern_task(min_val=min_val, max_val=max_val, rng=rng)
    elif task_type == "pattern_tuple":
        task = generate_pattern_tuple_task(
            digit_min=digit_min, digit_max=digit_max, rng=rng
        )
    elif task_type == "progression":
        # Need room for A+2 and B+2 within range.
        hi = max_val - 2
        lo = min_val
        if hi < lo:
            raise ValueError(
                f"progression ICL needs max_val-2 >= min_val "
                f"(got min={min_val}, max={max_val})"
            )
        b = rng.randint(lo, hi)
        a = rng.randint(lo, hi)
        correct = b + 2
        distractors = generate_distractors(correct, min_val, max_val, rng=rng)
        options = [correct] + distractors
        rng.shuffle(options)
        return {
            "task_type": "progression",
            "matrix": [[a, a + 1, a + 2], [b, b + 1, b + 2], [b, b + 1, None]],
            "answer_options": options,
            "correct_index": options.index(correct),
        }
    elif task_type == "combine":
        # combine uses digit_min>=1 by default
        return generate_combine_task(
            digit_min=max(1, digit_min), digit_max=digit_max, rng=rng
        )
    elif task_type == "intersection":
        return generate_intersection_task(
            digit_min=max(1, digit_min), digit_max=digit_max, rng=rng
        )
    elif task_type == "distribution_of_three":
        return generate_distribution_of_three_task(
            min_val=min_val, max_val=max_val, rng=rng
        )
    elif task_type == "progression_plus_n":
        step = rng.choice(PROGRESSION_STEPS)
        return generate_progression_plus_n_task(
            step=step, min_val=min_val, max_val=max_val, rng=rng
        )
    elif task_type == "tuple_grid":
        return generate_tuple_grid_task(
            min_val=min_val, max_val=max_val, rng=rng
        )
    else:
        raise ValueError(f"unknown task type: {task_type!r}")

    return _expand_task_to_3x3(task)


def task_fingerprint(task: dict) -> str:
    """Hashable identity for overlap checks (matrix + options + correct index)."""
    import json

    return json.dumps(
        {
            "matrix": task["matrix"],
            "answer_options": task["answer_options"],
            "correct_index": task["correct_index"],
        },
        sort_keys=True,
    )


def matrix_fingerprint(task: dict) -> str:
    """Matrix-only identity (exclude same grid even with different options)."""
    import json

    return json.dumps(task["matrix"], sort_keys=True)


def generate_in_context_examples(
    exclude: set[str],
    *,
    exclude_matrices: Optional[set[str]] = None,
    n_per_type: int = 3,
    seed: int = 20260619,
    max_attempts: int = 5000,
    min_val: int = 1,
    max_val: int = 20,
    digit_min: int = 0,
    digit_max: int = 9,
) -> dict[str, list[dict]]:
    """Generate ICL demos that follow task patterns but are not in ``exclude``.

    ``min_val`` / ``max_val`` widen the pool for magnitude types when many
    unique demos are needed (e.g. per-task ICL). ``digit_min`` / ``digit_max``
    do the same for digit/tuple types.
    """
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {tt: [] for tt in _ICL_TASK_TYPES}
    seen = set(exclude)
    seen_matrices = set(exclude_matrices or ())
    # Scale attempts with demand; digit types may need many retries.
    attempts_cap = max(max_attempts, n_per_type * 200)

    for task_type in _ICL_TASK_TYPES:
        attempts = 0
        while len(out[task_type]) < n_per_type and attempts < attempts_cap:
            attempts += 1
            task = _generate_icl_candidate(
                task_type,
                rng,
                min_val=min_val,
                max_val=max_val,
                digit_min=digit_min,
                digit_max=digit_max,
            )
            fp = task_fingerprint(task)
            mf = matrix_fingerprint(task)
            if fp in seen or mf in seen_matrices:
                continue
            seen.add(fp)
            seen_matrices.add(mf)
            out[task_type].append(
                {
                    "matrix": task["matrix"],
                    "answer_options": task["answer_options"],
                    "correct_letter": LETTERS[task["correct_index"]],
                    "correct_index": task["correct_index"],
                    "task_type": task_type,
                }
            )
        if len(out[task_type]) < n_per_type:
            raise RuntimeError(
                f"Could only generate {len(out[task_type])}/{n_per_type} "
                f"ICL examples for {task_type} after {attempts} attempts"
            )
    return out
