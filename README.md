# Raven's Numerical Reasoning Tasks

A benchmark and toolkit for Raven's-style numerical matrix reasoning: task generation, validation, Ollama evaluation, and a **baby-reasoning** harness (local vLLM/Ollama + [Modal](baby_reasoning_eval/modal_eval.py) GPU eval).

## Repository setup

```bash
git clone git@github.com:Ashley-Malkin/ravens-numerical-tasks.git
cd ravens-numerical-tasks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e baby_reasoning_eval/baby-reasoning
pip install modal   # optional: cloud GPU eval
```

- **Task generation & Ollama eval** — this directory (`generate.py`, `evaluate.py`, `tasks.json`).
- **Pythia / Qwen3 harness** — [`baby_reasoning_eval/README.md`](baby_reasoning_eval/README.md) (vLLM, Ollama, Modal).
- **Experiment log** — [`baby_reasoning_eval/experiments.md`](baby_reasoning_eval/experiments.md) (updated by Modal entrypoint runs).

### Attribution

[`baby_reasoning_eval/baby-reasoning/`](baby_reasoning_eval/baby-reasoning/) is vendored from [Ashley-Malkin/baby-reasoning](https://github.com/Ashley-Malkin/baby-reasoning) (MIT; see [`baby-reasoning/LICENSE`](baby_reasoning_eval/baby-reasoning/LICENSE)) with Ravens-specific extensions in this repo.

---

A CLI tool that generates numerical reasoning tasks inspired by Raven's Standard Progressive Matrices. Produces JSON output with 2x2 matrices and multiple-choice answer sets.

## Task Types

### 1. Numerical Constancy Matching

All matrix positions show the same number except the last cell, which is blank.

```
| N | N |
| N | ? |   → correct answer: N
```

- **Matrix:** (0,0), (0,1), (1,0) = N; (1,1) = blank
- **Correct answer:** N
- **Distractors:** 3 nearby numbers (e.g., N±1, N±2)

### 2. Pattern Following

The left column is constant; the right column has one known value and one blank.

```
| X | Y |
| X | ? |   → correct answer: Y (column stays constant)
```

- **Matrix:** (0,0)=X, (1,0)=X, (0,1)=Y (Y≠X); (1,1)=blank
- **Correct answer:** Y (value at position (0,1))
- **Distractors:** 3 nearby numbers to Y

### 3. Pattern Following (Tuple)

Tests pattern matching with two variables: digit values and number of digits. Each cell contains a tuple of digits (0-9).

```
| A    | B     |
| B    | ?     |   → correct answer: A (diagonal pattern)
```

Example: `[[(2), (3,0)],[(3,0), (null)]]` → correct: `(2)`

- **Matrix:** (0,0)=A, (0,1)=B, (1,0)=B, (1,1)=blank. A and B differ in digit values and/or length.
- **Correct answer:** A (diagonal: (0,0) matches (1,1))
- **Distractors:** Vary by both digit values (same length, different digits) and number of digits (different length)

### 4. Progression

Second column is one higher than the first column in each row.

```
| A   | A+1  |
| B   | ?    |   → correct answer: B + 1
```

- **Matrix:** (0,0)=A, (0,1)=A+1, (1,0)=B, (1,1)=blank
- **Correct answer:** B + 1 (second column = first column + 1)
- **Distractors:** 3 nearby numbers to B+1

### 5. Logic AND (3x3)

Third column is the tuple of both elements from columns 1 and 2 in each row.

```
| [1]   | [2]   | [1,2]  |
| [4]   | [5]   | [4,5]  |
| [3]   | [4]   | ?      |   → correct answer: [3, 4]
```

- **Matrix:** 3x3; each row: col3 = (col1, col2)
- **Correct answer:** Tuple with both elements from row 2

### 6. Logic OR tuple (3x3)

Each row has two **pairs** (length-2 lists) that share **exactly one** element. The third column is that shared element as a one-element list. The blank is the shared element for the last row.

```
| (3,4) | (8,4) | [4]   |
| (2,9) | (9,4) | [9]   |
| (2,0) | (9,2) | ?     |   → correct answer: 2
```

- **Matrix:** 3×3; cells in columns 1–2 are pairs; column 3 is `[x]` or `null` for the query row.
- **Rule:** In each row, `|col1 ∩ col2| = 1`; column 3 is that intersection.
- **Answer options:** Four **integers**: the correct shared element plus three distractors. Distractors are any other values in the generator’s digit range (`--min` / `--max`), chosen the same way as constancy/pattern (nearby numbers, then fill from the range).

## Usage

```bash
python generate.py --count 50 --type both --min 1 --max 20 --output tasks.json
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--count` | Number of tasks to generate | 10 |
| `--type` | `constancy`, `pattern`, `pattern_tuple`, `progression`, `combine`, `intersection`, `both`, or `all` | both |
| `--min` | Minimum number in matrix range | 1 |
| `--max` | Maximum number in matrix range | 20 |
| `--output` | Output JSON file path | tasks.json |
| `--seed` | Random seed for reproducibility | (none) |

## Output Format

```json
{
  "tasks": [
    {
      "task_type": "constancy",
      "matrix": [[5, 5], [5, null]],
      "answer_options": [3, 5, 4, 6],
      "correct_index": 1
    },
    {
      "task_type": "pattern",
      "matrix": [[7, 12], [7, null]],
      "answer_options": [11, 13, 12, 10],
      "correct_index": 2
    },
    {
      "task_type": "pattern_tuple",
      "matrix": [[[2], [3, 0]], [[3, 0], null]],
      "answer_options": [[1], [2], [3], [2, 0]],
      "correct_index": 1
    },
    {
      "task_type": "progression",
      "matrix": [[5, 6], [12, null]],
      "answer_options": [11, 14, 13, 12],
      "correct_index": 2
    },
    {
      "task_type": "combine",
      "matrix": [[[1],[2],[1,2]], [[4],[5],[4,5]], [[3],[4],null]],
      "answer_options": [[2,3], [3,4], [1,4], [4,3]],
      "correct_index": 1
    },
    {
      "task_type": "intersection",
      "matrix": [
        [[3, 4], [8, 4], [4]],
        [[2, 9], [9, 4], [9]],
        [[2, 0], [9, 2], null]
      ],
      "answer_options": [2, 5, 0, 3],
      "correct_index": 0
    }
  ]
}
```

- `matrix`: 2x2 or 3x3 array; `null` denotes the blank cell
- `answer_options`: 4 shuffled choices (1 correct + 3 distractors)
- `correct_index`: index (0–3) of the correct answer in `answer_options`

## Evaluation

To evaluate a model (e.g. Qwen3 30B) via Ollama on the task set:

1. Install dependencies: `pip install requests`
2. Run Ollama locally with the model: `ollama run qwen3:30b` (or pull first)
3. Run the evaluator:

```bash
python evaluate.py --tasks tasks.json --model qwen3:30b
# Structured JSON answer (Ollama `format` + `think: false` + logprobs for A–D):
python evaluate.py --tasks tasks.json --model qwen3:30b --choice-only
```

Options: `--limit N` (run first N tasks), `--n-examples N` (0/1/3 in-context demos), `--base-url URL`, `--timeout SEC`, `--verbose`, `--choice-only` (JSON `{"choice":"A"|…}` plus logprobs at the **choice-letter token**; summary includes argmax accuracy and **mean Brier score** from softmax p(A–D) vs one-hot correct), `--cot-choice`, `--debug`, `--no-thinking`, `--max-tokens N`.

## One-off migration scripts

Historical scripts used to build or fix `tasks.json` live under [`scripts/migrations/`](scripts/migrations/).

## Requirements

Python 3.9+. Generation uses stdlib only; evaluation requires `requests`.
