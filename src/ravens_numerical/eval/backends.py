from __future__ import annotations

import re

import requests

from ravens_numerical.eval.tasks.base import ModelBackend, ModelResponse
from ravens_numerical.models.registry import is_qwen3_instruct_model, max_model_len_for_model
from ravens_numerical.scoring.choice_logprobs import CHOICE_ONLY_FORMAT
from ravens_numerical.scoring.mlm_scoring import score_completion_span, score_sequence

CHOICE_ONLY_MAX_TOKENS = 256


def strip_qwen_thinking(text: str) -> str:
    """Remove Qwen3 reasoning blocks that may appear in completion output."""
    for pat in (
        r"<\s*think\s*>.*?<\s*/\s*think\s*>",
        r"<\s*redacted_reasoning\s*>.*?<\s*/\s*redacted_reasoning\s*>",
    ):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class OllamaBackend(ModelBackend):
    """ModelBackend for Ollama's ``/api/generate`` (e.g. Qwen3), matching ``evaluate.call_ollama``."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_tokens: int = 64,
    ) -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def _post_generate(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        _ = kwargs
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.max_tokens,
            },
        }
        data = self._post_generate(payload)
        text = (data.get("response") or "").strip()
        thinking_text = (data.get("thinking") or "").strip()
        if not text and thinking_text:
            text = thinking_text
        return ModelResponse(text=text.rstrip(), token_logprobs=None)

    def score_completion(self, prompt: str, completion: str) -> float | None:
        """Ollama generate API does not expose echo+logprob scoring like vLLM."""
        return None


class VLLMBackend(ModelBackend):
    """ModelBackend implementation over the vLLM OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000") -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self._is_qwen3_instruct = is_qwen3_instruct_model(model)

    @property
    def model(self) -> str:
        return self._model

    def _post(self, path: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        _ = kwargs
        data = self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0,
                "logprobs": 1,
            },
        )
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs")
        token_logprobs = (
            logprobs_data.get("token_logprobs")
            if isinstance(logprobs_data, dict)
            else None
        )
        text = choice.get("text", "").rstrip()
        if self._is_qwen3_instruct:
            text = strip_qwen_thinking(text)
        return ModelResponse(
            text=text,
            token_logprobs=token_logprobs,
        )

    def score_completion(self, prompt: str, completion: str) -> float | None:
        """Return sum of token log probs for the completion only, or None if unsupported."""
        data = self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt + completion,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
            },
        )
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs")
        if not isinstance(logprobs_data, dict):
            return None
        token_logprobs = logprobs_data.get("token_logprobs")
        if token_logprobs is None:
            return None
        text_offset = logprobs_data.get("text_offset")
        if text_offset is None:
            return None
        prompt_len = len(prompt)
        return sum(
            lp for lp, off in zip(token_logprobs, text_offset)
            if off >= prompt_len and lp is not None
        )


class ChoiceOnlyVLLMBackend(ModelBackend):
    """Instruction choice-only eval via vLLM structured JSON + choice-token logprobs."""

    _api_path = "/v1/completions"

    def __init__(self, model: str, base_url: str = "http://localhost:8000") -> None:
        self._model = model
        self.base_url = base_url.rstrip("/")
        self._choice_schema = CHOICE_ONLY_FORMAT

    @property
    def model(self) -> str:
        return self._model

    def _post(self, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}{self._api_path}",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _response_format_payload(self) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "choice_only",
                "schema": self._choice_schema,
                "strict": True,
            },
        }

    def _build_generate_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": CHOICE_ONLY_MAX_TOKENS,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "extra_body": {"guided_json": self._choice_schema},
        }

    def _extract_text(self, data: dict) -> str:
        choice = data["choices"][0]
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content).strip()
        return str(choice.get("text", "")).strip()

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        _ = kwargs
        data = self._post(self._build_generate_payload(prompt))
        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs")
        token_logprobs = None
        if isinstance(logprobs_data, dict):
            token_logprobs = logprobs_data.get("token_logprobs")
        return ModelResponse(
            text=self._extract_text(data),
            token_logprobs=token_logprobs,
            raw=data,
        )

    def score_completion(self, prompt: str, completion: str) -> float | None:
        return None


class PythiaChoiceOnlyVLLMBackend(ChoiceOnlyVLLMBackend):
    """GPT-2 style instruction (Pythia, BabyLM): completions API + guided JSON."""


class Qwen3ChoiceOnlyVLLMBackend(ChoiceOnlyVLLMBackend):
    """Qwen3 instruction: chat completions + JSON schema + thinking off."""

    _api_path = "/v1/chat/completions"

    def _build_generate_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": CHOICE_ONLY_MAX_TOKENS,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "response_format": self._response_format_payload(),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }


def _choice_suffix(letter: str, *, cot: bool = False) -> str:
    if cot:
        return f'\n{{"reasoning":"","choice":"{letter}"}}'
    return f'\n{{"choice":"{letter}"}}'


def _build_synthetic_choice_raw(text: str, logprobs_by_letter: dict[str, float]) -> dict:
    """vLLM-shaped response so ``parse_logprobs_by_letter_vllm`` works."""
    m = re.search(r'(?i)"choice"\s*:\s*"\s*([ABCD])', text.strip())
    chosen = m.group(1).upper() if m else "A"
    prefix = text[: m.start(1)] if m else '{"choice":"'
    suffix = text[m.end(1) :] if m else '"}'
    tokens = [prefix, chosen, suffix]
    lps = [0.0, logprobs_by_letter.get(chosen, 0.0), 0.0]
    top_logprobs: list = [None, {ltr: logprobs_by_letter[ltr] for ltr in "ABCD"}, None]
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": lps,
                    "top_logprobs": top_logprobs,
                },
            }
        ]
    }


class RobertaMLMBackend(ModelBackend):
    """RoBERTa masked LM (MiniBERTa): PLL scoring via HuggingFace transformers."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        prompt_type: str = "instruction",
        prompt_mode: str = "choice_only",
    ) -> None:
        self._model_id = model
        self._device = device
        self._prompt_type = prompt_type
        self._prompt_mode = prompt_mode
        self._model = None
        self._tokenizer = None
        self._max_length = max_model_len_for_model(model)

    @property
    def model(self) -> str:
        return self._model_id

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForMaskedLM.from_pretrained(self._model_id)
        if getattr(self._model.config, "mask_token_id", None) is None:
            mask_id = self._tokenizer.mask_token_id
            if mask_id is None:
                raise ValueError(f"No mask token for {self._model_id}")
            self._model.config.mask_token_id = mask_id
        if self._device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")
        else:
            self._device = "cpu"
            self._model = self._model.to("cpu")
        self._model.eval()

    def score_completion(self, prompt: str, completion: str) -> float | None:
        self._ensure_loaded()
        return score_completion_span(
            self._model,
            self._tokenizer,
            prompt,
            completion,
            max_length=self._max_length,
        )

    def _score_instruction_letters(self, prompt: str) -> tuple[str, dict[str, float]]:
        self._ensure_loaded()
        cot = self._prompt_mode == "cot_choice"
        scores: dict[str, float] = {}
        for letter in "ABCD":
            suffix = _choice_suffix(letter, cot=cot)
            scores[letter] = score_sequence(
                self._model,
                self._tokenizer,
                prompt + suffix,
                max_length=self._max_length,
            )
        best = max("ABCD", key=lambda c: scores[c])
        if cot:
            text = f'{{"reasoning":"","choice":"{best}"}}'
        else:
            text = f'{{"choice":"{best}"}}'
        return text, scores

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        stimulus = kwargs.get("stimulus")
        task = kwargs.get("task")

        if self._prompt_type == "completion" and stimulus is not None and task is not None:
            choices = stimulus.answer_choices or []
            if not choices:
                return ModelResponse(text="", token_logprobs=None)
            scored = {
                c: self.score_completion(prompt, task.format_completion(stimulus, c))
                for c in choices
            }
            valid = {c: s for c, s in scored.items() if s is not None}
            best = max(valid, key=valid.get) if valid else choices[0]
            return ModelResponse(text=best.split("]")[0].strip(), token_logprobs=None)

        if self._prompt_type == "instruction" and self._prompt_mode == "choice_only":
            text, letter_scores = self._score_instruction_letters(prompt)
            raw = _build_synthetic_choice_raw(text, letter_scores)
            return ModelResponse(text=text, token_logprobs=None, raw=raw)

        if self._prompt_type == "instruction":
            text, _ = self._score_instruction_letters(prompt)
            m = re.search(r'(?i)"choice"\s*:\s*"\s*([ABCD])', text)
            letter = m.group(1).upper() if m else "A"
            if self._prompt_mode == "plain":
                return ModelResponse(text=letter, token_logprobs=None)
            return ModelResponse(text=text, token_logprobs=None)

        return ModelResponse(text="", token_logprobs=None)
