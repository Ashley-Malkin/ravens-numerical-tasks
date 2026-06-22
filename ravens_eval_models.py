"""Default model IDs aligned with ``evaluate.py`` and ``baby_reasoning_eval``.

Use these with ``baby_reasoning/script/run``: ``--backend vllm`` + HF ids, or ``--backend ollama``
+ tags below.
"""

# Same tags as ``evaluate.DEBUG_MODELS`` (Ollama).
QWEN3_OLLAMA_MODELS: tuple[str, ...] = (
    "qwen3:30b",
    "qwen3:8b",
    "qwen3:4b",
    "qwen3:1.7b",
    "qwen3:0.6b",
)

# Typical small Pythia checkpoint for vLLM (match baby-reasoning ``do_eval.sh`` style).
DEFAULT_PYTHIA_VLLM = "EleutherAI/pythia-70m-deduped"
