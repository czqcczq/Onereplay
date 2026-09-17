"""Shared generation helpers for OneReplay evaluation.

Two decoding paths live here. The single-prompt one (generate_response and
friends) is the reference implementation and stays around for debugging and for
multi-turn metrics whose next input depends on the previous output. Everything
that decodes a fixed prompt list should use the batched path: at batch=1 an 8B
model reaches maybe a tenth of an H200's memory bandwidth, so a full GSM8K pass
costs hours instead of minutes.

The two paths are not bit-identical -- batching adds left padding and takes
different kernels -- so a mix of old and new numbers is not comparable. Every
arm of a comparison has to be decoded the same way. configure_decoding sets
that shared policy once per run; the default stays plain greedy.
"""

from __future__ import annotations

from typing import Any

import torch

from onereplay.core.chat_policy import current_system_prompt, with_system

DEFAULT_EVAL_BATCH_SIZE = 32

# Decoding is a property of the whole run, not of one metric: a comparison that
# mixes greedy and sampled arms is meaningless, and a per-call argument is one
# every metric would have to remember to forward. So it lives here as run-wide
# state that evaluate.py sets once, before any metric runs.
_DECODE_CONFIG: dict[str, Any] = {}
# Every default in this module reproduces the behavior from before it existed,
# so callers that never touch configure_decoding -- replay generation, the CE
# probes -- keep producing numbers comparable with their own history.
_STOP_ON_IM_END = False


def configure_decoding(*, stop_on_im_end: bool = False, **overrides: Any) -> None:
    """Replace the run-wide generate() overrides. Falsy overrides are dropped."""

    global _STOP_ON_IM_END
    _STOP_ON_IM_END = bool(stop_on_im_end)
    _DECODE_CONFIG.clear()
    _DECODE_CONFIG.update({key: value for key, value in overrides.items() if value})


def describe_decoding() -> str:
    """One-line decoding summary, for the run log."""

    parts = [f"{key}={value}" for key, value in sorted(_DECODE_CONFIG.items())]
    if not _DECODE_CONFIG.get("do_sample"):
        parts.insert(0, "greedy")
    if _STOP_ON_IM_END:
        parts.append("stop_on_im_end=True")
    system = current_system_prompt()
    parts.append(f"system={system!r}" if system else "system=<template default>")
    return " ".join(parts)


def stop_token_ids(tokenizer, stop_on_im_end: bool = False) -> list[int] | None:
    """EOS ids for generate().

    The Qwen3 *base* tokenizer reports eos_token='<|endoftext|>', but a trained
    chat turn ends '<|im_end|><|endoftext|>'. Stopping on eos_token alone lets a
    model that already emitted a perfectly good '<|im_end|>' run on for
    thousands of tokens into a hallucinated next turn, which skip_special_tokens
    then splices onto the answer.
    """

    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if stop_on_im_end:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end not in ids:
            ids.append(im_end)
    return ids or None


def _generate_kwargs(tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_token_ids(tokenizer, _STOP_ON_IM_END),
    }
    kwargs.update(_DECODE_CONFIG)
    return kwargs


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a conversation with an open assistant turn."""

    messages = with_system(messages)
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
            **_generate_kwargs(tokenizer),
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
                **_generate_kwargs(tokenizer),
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
