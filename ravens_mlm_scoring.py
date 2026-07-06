"""Pseudo-log-likelihood scoring for RoBERTa / MiniBERTa masked LMs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def encode_tail_truncated(
    tokenizer: Any,
    text: str,
    *,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize ``text``, keeping the tail if longer than ``max_length``."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return enc["input_ids"], enc["attention_mask"]


def _special_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            ids.add(int(val))
    return ids


def mask_token_id(model: Any, tokenizer: Any | None = None) -> int:
    """Resolve RoBERTa ``<mask>`` id (nyu-mll configs often omit ``mask_token_id``)."""
    if tokenizer is not None:
        tok_id = getattr(tokenizer, "mask_token_id", None)
        if tok_id is not None:
            return int(tok_id)
    cfg_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    if cfg_id is not None:
        return int(cfg_id)
    raise ValueError("Could not resolve mask_token_id from model config or tokenizer")


def pseudo_log_likelihood(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    tokenizer: Any | None = None,
    start: int = 0,
    end: int | None = None,
) -> float:
    """Sum log-probs of tokens in ``[start, end)`` via one-token masking (PLL)."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    mask_id = mask_token_id(model, tokenizer)

    seq_len = int(input_ids.shape[1])
    end = seq_len if end is None else min(end, seq_len)
    total = 0.0
    n_scored = 0

    with torch.no_grad():
        for idx in range(start, end):
            if int(attention_mask[0, idx].item()) == 0:
                continue
            masked = input_ids.clone()
            original = int(masked[0, idx].item())
            if original == mask_id:
                continue
            masked[0, idx] = mask_id
            logits = model(input_ids=masked, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[0, idx], dim=-1)
            total += float(log_probs[original].item())
            n_scored += 1

    if n_scored == 0:
        return float("-inf")
    return total


def score_completion_span(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int = 512,
) -> float:
    """PLL over completion tokens in ``prompt + completion`` (tail-truncated)."""
    full = prompt + completion
    input_ids, attention_mask = encode_tail_truncated(
        tokenizer, full, max_length=max_length
    )
    prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=True))
    full_token_len = len(tokenizer.encode(full, add_special_tokens=True))
    completion_len = full_token_len - prompt_token_len
    seq_len = int(input_ids.shape[1])
    start = max(1, seq_len - completion_len)
    return pseudo_log_likelihood(
        model,
        input_ids,
        attention_mask,
        tokenizer=tokenizer,
        start=start,
        end=seq_len,
    )


def score_sequence(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_length: int = 512,
) -> float:
    """PLL over all non-special tokens in ``text``."""
    input_ids, attention_mask = encode_tail_truncated(
        tokenizer, text, max_length=max_length
    )
    special = _special_token_ids(tokenizer)
    seq_len = int(input_ids.shape[1])
    start = 1
    end = seq_len - 1 if seq_len > 1 else seq_len
    total = 0.0
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    mask_id = mask_token_id(model, tokenizer)

    with torch.no_grad():
        for idx in range(start, end):
            if int(attention_mask[0, idx].item()) == 0:
                continue
            original = int(input_ids[0, idx].item())
            if original in special:
                continue
            masked = input_ids.clone()
            masked[0, idx] = mask_id
            logits = model(input_ids=masked, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[0, idx], dim=-1)
            total += float(log_probs[original].item())

    return total
