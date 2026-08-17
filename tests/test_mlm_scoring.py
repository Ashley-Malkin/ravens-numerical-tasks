#!/usr/bin/env python3
"""Tests for MiniBERTa PLL scoring helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

from ravens_numerical.scoring.mlm_scoring import (
    completion_token_span,
    encode_tail_truncated,
    mask_token_id,
    pseudo_log_likelihood,
)


class _FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None, add_special_tokens=True):
        ids = list(range(min(len(text.split()), max_length or 512)))
        if not ids:
            ids = [1, 2]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=True):
        return list(range(min(len(text.split()), 512)))


def test_encode_tail_truncated_respects_max_length():
    tok = _FakeTokenizer()
    long_text = "word " * 400
    ids, _ = encode_tail_truncated(tok, long_text, max_length=64)
    assert ids.shape[1] <= 64


class _TinyMLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"mask_token_id": 3})()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        logits = self.head(self.embed(input_ids))
        return type("Out", (), {"logits": logits})()


def test_encode_tail_truncated_keeps_tail():
    test_encode_tail_truncated_respects_max_length()


def test_mask_token_id_from_tokenizer_when_config_lacks_attr():
    model = type("M", (), {"config": type("Cfg", (), {})()})()
    tok = _FakeTokenizer()
    assert mask_token_id(model, tok) == 3


def test_completion_token_span_excludes_cls_and_sep():
    # [CLS] + 20 prompt + c completion + [SEP]
    prompt_toks = 20
    for c in (1, 2, 3, 4):
        seq_len = 1 + prompt_toks + c + 1
        start, end = completion_token_span(seq_len, c)
        assert start == 1 + prompt_toks
        assert end == seq_len - 1
        assert end - start == c


def test_completion_token_span_old_formula_was_misaligned():
    """Regression: ``seq_len - completion_len`` skipped first token and scored SEP."""
    prompt_toks, c = 20, 2
    seq_len = 1 + prompt_toks + c + 1
    old_start = max(1, seq_len - c)
    old_end = seq_len
    start, end = completion_token_span(seq_len, c)
    assert (old_start, old_end) != (start, end)
    assert start == old_start - 1
    assert end == old_end - 1


def test_pseudo_log_likelihood_runs():
    model = _TinyMLM()
    tok = _FakeTokenizer()
    input_ids = torch.tensor([[1, 5, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    score = pseudo_log_likelihood(
        model, input_ids, attention_mask, tokenizer=tok, start=1, end=3
    )
    assert isinstance(score, float)


def test_build_synthetic_choice_raw_parsable():
    from ravens_numerical.eval.backends import _build_synthetic_choice_raw
    from ravens_numerical.scoring.choice_logprobs import parse_logprobs_by_letter_vllm

    text = '{"choice":"C"}'
    scores = {"A": -2.0, "B": -1.5, "C": -0.5, "D": -3.0}
    raw = _build_synthetic_choice_raw(text, scores)
    pred, lps = parse_logprobs_by_letter_vllm(raw, text)
    assert pred == 2
    assert lps["C"] == -0.5


if __name__ == "__main__":
    test_encode_tail_truncated_keeps_tail()
    test_mask_token_id_from_tokenizer_when_config_lacks_attr()
    test_completion_token_span_excludes_cls_and_sep()
    test_completion_token_span_old_formula_was_misaligned()
    test_pseudo_log_likelihood_runs()
    test_build_synthetic_choice_raw_parsable()
    print("OK")
