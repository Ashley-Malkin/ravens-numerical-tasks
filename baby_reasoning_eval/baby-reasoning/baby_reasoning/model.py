from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

from baby_reasoning.tasks.base import ModelBackend, ModelResponse

CHOICE_ONLY_MAX_TOKENS = 256


def _repo_root() -> Path:
    # baby_reasoning/model.py -> baby_reasoning, clone root, baby_reasoning_eval, ravens root
    return Path(__file__).resolve().parents[3]


def _import_choice_logprobs():
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ravens_choice_logprobs import CHOICE_ONLY_FORMAT

    return CHOICE_ONLY_FORMAT


def _import_is_qwen3():
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ravens_eval_models import is_qwen3_model

    return is_qwen3_model


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

    def generate(self, prompt: str) -> ModelResponse:
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
        self._is_qwen3 = _import_is_qwen3()(model)

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

    def generate(self, prompt: str) -> ModelResponse:
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
        if self._is_qwen3:
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
        self._choice_schema = _import_choice_logprobs()

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

    def generate(self, prompt: str) -> ModelResponse:
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
    """Pythia instruction: completions API + guided JSON."""


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
