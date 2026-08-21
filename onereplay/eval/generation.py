"""Shared generation helpers for OneReplay evaluation."""

from __future__ import annotations

from typing import Any

import torch


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a conversation with an open assistant turn."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def render_user_prompt(tokenizer, prompt: str) -> str:
    """Format a single user prompt as a chat turn."""

    return render_chat(tokenizer, [{"role": "user", "content": prompt}])


def generate_from_text(
    model,
    tokenizer,
    text: str,
    device,
    max_new_tokens: int,
) -> str:
    """Greedy-decode one continuation for already chat-rendered text."""

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int,
) -> str:
    """Greedy-decode one assistant response for a single-turn prompt."""

    return generate_from_text(
        model, tokenizer, render_user_prompt(tokenizer, prompt), device, max_new_tokens
    )


def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    device,
    max_new_tokens: int,
) -> str:
    """Greedy-decode one assistant response for a multi-turn conversation."""

    return generate_from_text(
        model, tokenizer, render_chat(tokenizer, messages), device, max_new_tokens
    )
