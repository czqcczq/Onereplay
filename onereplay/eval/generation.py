"""Shared generation helpers for OneReplay evaluation.

Two decoding paths live here. The single-prompt one (generate_response and
friends) is the reference implementation and stays around for debugging and for
multi-turn metrics whose next input depends on the previous output. Everything
that decodes a fixed prompt list should use the batched path: at batch=1 an 8B
model reaches maybe a tenth of an H200's memory bandwidth, so a full GSM8K pass
costs hours instead of minutes.

The two paths are not bit-identical -- batching adds left padding and takes
different kernels -- so a mix of old and new numbers is not comparable. Every
arm of a comparison has to be decoded the same way.
"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_EVAL_BATCH_SIZE = 32


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
    strip: bool = True,
) -> str:
    """Greedy-decode one continuation for already chat-rendered text.

    Pass ``strip=False`` when leading whitespace carries meaning. HumanEval
    completions are appended to a function stub, so stripping the first line's
    indentation turns a correct body into ``'return' outside function``.
    """

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
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip() if strip else decoded


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int,
    strip: bool = True,
) -> str:
    """Greedy-decode one assistant response for a single-turn prompt."""

    return generate_from_text(
        model,
        tokenizer,
        render_user_prompt(tokenizer, prompt),
        device,
        max_new_tokens,
        strip=strip,
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


def batched_generate_from_texts(
    model,
    tokenizer,
    texts: list[str],
    device,
    max_new_tokens: int,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    strip: bool = True,
    log_label: str = "",
    log_every: int = 5,
) -> list[str]:
    """Greedy-decode one continuation per text, batched and order-preserving.

    Texts are length-sorted into buckets so a batch is not dragged out by one
    long member, and each row keeps its original index, so the returned list
    lines up with ``texts`` regardless of the bucket it was decoded in.

    Left padding is required and comes from load_causal_lm_and_tokenizer; with
    right padding the generated tokens would follow the pad run instead of the
    prompt. Pass ``strip=False`` when leading whitespace carries meaning, as it
    does for HumanEval completions appended to a function stub.
    """

    if not texts:
        return []

    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    order = sorted(range(len(texts)), key=lambda index: lengths[index])

    responses: list[str] = [""] * len(texts)
    num_batches = (len(order) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(order), batch_size), start=1):
        chunk = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[index] for index in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[:, encoded["input_ids"].shape[1] :]
        for index, ids in zip(chunk, generated):
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            responses[index] = decoded.strip() if strip else decoded
        if log_label and log_every > 0 and batch_number % log_every == 0:
            print(
                f"{log_label} generated batch {batch_number}/{num_batches}",
                flush=True,
            )
    return responses


def batched_generate(
    model,
    tokenizer,
    prompts: list[str],
    device,
    max_new_tokens: int,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    strip: bool = True,
    log_label: str = "",
    log_every: int = 5,
) -> list[str]:
    """Greedy-decode one assistant response per single-turn prompt."""

    return batched_generate_from_texts(
        model,
        tokenizer,
        [render_user_prompt(tokenizer, prompt) for prompt in prompts],
        device,
        max_new_tokens,
        batch_size=batch_size,
        strip=strip,
        log_label=log_label,
        log_every=log_every,
    )


def resolve_batch_size(cfg: dict[str, Any], *keys: str) -> int:
    """Pick a decoding batch size from cfg, newest-specific key first.

    Metrics pass their own override key ahead of the shared eval_batch_size so
    a single job can, say, decode 4096-token math at 16 and 512-token code at
    64 without adding a flag per metric.
    """

    for key in (*keys, "eval_batch_size"):
        value = cfg.get(key)
        if value:
            return max(1, int(value))
    return DEFAULT_EVAL_BATCH_SIZE
