"""Chat template helpers and dataset collators for OneReplay training."""

from __future__ import annotations

from onereplay._imports import ensure_project_root

ensure_project_root()

from process_dataset.process_glue_myself import build_loader, tokenizer_to_ids  # noqa: E402


def apply_train_template(tokenizer, instruction: str, input_text: str, output_text: str) -> tuple[str, str]:
    """Return full training text and prompt-only text for label masking.

    full_text contains both user and assistant messages.
    prompt_text ends at the assistant generation point and excludes the answer.
    tokenizer_to_ids then masks prompt_text tokens with -100.
    """

    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\nInput:\n{input_text.strip()}"

    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text.strip()},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    ).rstrip()
    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return full_text, prompt_text


def build_sft_tokenize_fn(tokenizer, max_len: int):
    """Return a datasets.map function for {instruction, input, output} rows.

    Commonsense170k and the replay corpus both go through this, so replayed
    old-knowledge samples are masked and truncated exactly like new-task ones.
    """

    def add_tokenized_fields(example):
        full_text, prompt_text = apply_train_template(
            tokenizer,
            example["instruction"],
            example.get("input", ""),
            example["output"],
        )
        tokenized = tokenizer_to_ids(
            tokenizer=tokenizer,
            text=full_text,
            prompt_text=prompt_text,
            max_length=max_len,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }

    return add_tokenized_fields


def apply_prompt_template(tokenizer, user_content: str, enable_thinking: bool = False) -> str:
    """Render a user-only chat prompt with an open assistant turn.

    enable_thinking stays False by default so every existing caller (evaluation,
    replay generation) keeps its current behavior; only self-distillation that
    explicitly asks for reasoning traces turns it on.
    """

    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_opd_loader(dataset, tokenizer, batch_size, train=True):
    """DataLoader that keeps instruction/input strings for on-policy distillation."""

    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    tokenizer.padding_side = "left"
    token_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    keep_columns = ["input_ids", "labels", "attention_mask", "instruction", "input"]
    columns_to_remove = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(columns_to_remove)

    def collate_fn(examples):
        instructions = [example.get("instruction", "") for example in examples]
        inputs = [example.get("input", "") for example in examples]
        token_batch = token_collator(
            [
                {
                    "input_ids": example["input_ids"],
                    "labels": example["labels"],
                    "attention_mask": example["attention_mask"],
                }
                for example in examples
            ]
        )
        token_batch["instruction"] = instructions
        token_batch["input"] = inputs
        return token_batch

    return DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=train)


__all__ = [
    "apply_prompt_template",
    "apply_train_template",
    "build_loader",
    "build_opd_loader",
    "build_sft_tokenize_fn",
    "tokenizer_to_ids",
]
