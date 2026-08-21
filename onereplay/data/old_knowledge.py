"""Old-knowledge corpus loading (FLAN) for covariance collection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk


def load_old_knowledge_dataset(args: argparse.Namespace):
    """Load the old-knowledge corpus from disk, local files, or HuggingFace."""

    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
        return dataset[args.dataset_split] if args.dataset_split in dataset else dataset

    if args.data_files:
        suffix = Path(args.data_files).suffix.lower()
        file_format = "text" if suffix == ".txt" else "json"
        dataset = load_dataset(
            file_format,
            data_files=args.data_files,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
        )
        return dataset

    config = args.dataset_config if args.dataset_config else None
    return load_dataset(
        args.dataset_name,
        config,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
    )


def filter_incomplete_rows(dataset, args: argparse.Namespace):
    """Drop rows whose input or target is empty.

    Needed to keep C on exactly the rows replay trains on. The self-distilled
    corpus stores an empty target for every prompt whose generation hit the
    token cap (2565 of 20000 at max_new_tokens=384), and the replay loader drops
    those rows via --replay_drop_truncated plus to_sft_schema's non-empty
    filter. Without the same filter here, C would still absorb the prompt-side
    activations of rows replay never sees.
    """

    input_column = args.input_column
    target_column = args.target_column

    def is_complete(example: dict[str, Any]) -> bool:
        target = example.get(target_column)
        if not (target and str(target).strip()):
            return False
        if args.text_column:
            source = example.get(args.text_column)
        else:
            source = example.get(input_column)
        return bool(source and str(source).strip())

    before = len(dataset) if hasattr(dataset, "__len__") else None
    dataset = dataset.filter(is_complete)
    if before is not None:
        after = len(dataset)
        print(
            f"Stage 1: require_target dropped {before - after} of {before} rows "
            f"with an empty {target_column} or {args.text_column or input_column}"
        )
    else:
        print(f"Stage 1: require_target filtering a streaming dataset on {target_column}")
    return dataset


def limit_dataset(dataset, args: argparse.Namespace):
    """Take a reproducible subset of a map-style Dataset or streaming IterableDataset.

    With --sample_shuffle 1 (default), the dataset is shuffled with
    --sample_seed before taking --max_samples, so the subset is a reproducible
    random sample of the corpus rather than its first N rows. Growing
    max_samples with the same seed keeps smaller subsets nested in larger ones
    for map-style datasets.

    Streaming datasets only support approximate buffer shuffling; the result is
    still deterministic for a fixed seed and buffer size.
    """

    max_samples = args.max_samples
    shuffle = args.sample_shuffle == 1

    if hasattr(dataset, "select"):
        if shuffle:
            dataset = dataset.shuffle(seed=args.sample_seed)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    if hasattr(dataset, "take"):
        if shuffle:
            dataset = dataset.shuffle(
                seed=args.sample_seed,
                buffer_size=args.shuffle_buffer_size,
            )
        if max_samples > 0:
            dataset = dataset.take(max_samples)
        return dataset

    return dataset


def fingerprint_pool(dataset, args: argparse.Namespace) -> tuple[int, str]:
    """Hash the selected rows so two stages can prove they saw the same corpus.

    C and F have to be estimated on the same old-knowledge rows or the OneReplay
    versus EWC comparison is confounded by which data each method got. The
    selection is deterministic on the map-style path (a full shuffle at a fixed
    seed followed by select), but "should be deterministic" and "was
    deterministic" are different claims, and a divergence caused by a library
    upgrade or an edited flag would not raise anything. Hashing the rows turns
    the assumption into a value both stages print and store.

    Streaming datasets are skipped rather than consumed: iterating an
    IterableDataset here would drain the very iterator the caller is about to
    forward through the model.
    """

    if not hasattr(dataset, "select"):
        return 0, "streaming-not-fingerprinted"

    digest = hashlib.sha256()
    rows = 0
    for example in dataset:
        if args.text_column:
            source = example.get(args.text_column)
        else:
            source = example.get(args.input_column)
        digest.update(str(source or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(example.get(args.target_column) or "").encode("utf-8"))
        digest.update(b"\x01")
        rows += 1
    return rows, digest.hexdigest()


def example_to_plain_text(example: dict[str, Any], args: argparse.Namespace) -> str:
    """Convert one FLAN-style row into plain text.

    This is kept for ablations. The default path below uses chat template,
    because Qwen instruction tuning and your BoolQ data are chat-formatted.
    """

    if args.text_column and args.text_column in example:
        return str(example[args.text_column])

    pieces: list[str] = []
    if args.input_column in example and example[args.input_column] is not None:
        pieces.append(str(example[args.input_column]))
    if args.target_column in example and example[args.target_column] is not None:
        pieces.append(str(example[args.target_column]))

    if pieces:
        return "\n".join(piece.strip() for piece in pieces if piece.strip())

    # Fallback for unknown schemas: join string-like fields so the script still
    # works with many json/jsonl datasets without code edits.
    for value in example.values():
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    return "\n".join(pieces)


def example_to_messages(example: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    """Convert one FLAN row into chat messages.

    FLAN has an instruction/input field and a target field. We map them to:

        user:      inputs
        assistant: targets

    This makes the old-knowledge hidden states live in the same chat-template
    distribution as the later LoRA fine-tuning data.
    """

    user_content = ""
    if args.text_column and args.text_column in example:
        user_content = str(example[args.text_column]).strip()
    elif args.input_column in example and example[args.input_column] is not None:
        user_content = str(example[args.input_column]).strip()
    else:
        user_content = example_to_plain_text(example, args).strip()

    messages: list[dict[str, str]] = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    target = ""
    if args.target_column in example and example[args.target_column] is not None:
        target = str(example[args.target_column]).strip()
    if args.include_target_in_chat == 1 and target:
        messages.append({"role": "assistant", "content": target})

    return messages


def example_to_model_text(example: dict[str, Any], tokenizer, args: argparse.Namespace) -> str:
    """Build the exact text that will be tokenized and forwarded through model.

    When use_chat_template=1, tokenizer.apply_chat_template inserts model-
    specific role tokens such as Qwen's user/assistant markers. If a tokenizer
    has no chat template, we fall back to plain text with a clear error-free path
    so the script still works for non-chat base models.
    """

    if args.use_chat_template != 1:
        return example_to_plain_text(example, args)

    messages = example_to_messages(example, args)
    if getattr(tokenizer, "chat_template", None) is None:
        return example_to_plain_text(example, args)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def example_to_prompt_text(example: dict[str, Any], tokenizer, args: argparse.Namespace) -> str:
    """Render the same row up to the point where the assistant would answer.

    Pairs with example_to_model_text for Fisher collection: the token prefix the
    two renders share is exactly the span that must not be supervised, because
    the Fisher of an instruction-following model should measure the sensitivity
    of its responses, not of the prompts it was handed.

    Callers compare token ids rather than trusting this string's length, so a
    chat template that inserts extra tokens into the generation prompt degrades
    into a shorter mask instead of a misaligned one.
    """

    if args.use_chat_template != 1 or getattr(tokenizer, "chat_template", None) is None:
        # Plain-text ablation. example_to_plain_text joins the input and the
        # target with a newline, so the prompt is everything before that join.
        if args.text_column and args.text_column in example:
            return str(example[args.text_column]).strip()
        source = example.get(args.input_column)
        return f"{str(source or '').strip()}\n"

    messages = [
        message for message in example_to_messages(example, args) if message["role"] != "assistant"
    ]
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


def build_collate_fn(tokenizer, args: argparse.Namespace):
    """Create a DataLoader collator that tokenizes text and pads each batch."""

    # Truncation drops the assistant turn by default (transformers truncates on
    # the right), while training keeps it (`full_input_ids[-max_length:]` in
    # tokenizer_to_ids drops the prompt head instead). With FLAN's one-line gold
    # targets almost nothing overflows, so the mismatch never mattered; with
    # self-distilled answers up to 384 tokens it decides whether C sees the
    # answers at all. Set --truncation_side left to match training.
    truncation_side = getattr(args, "truncation_side", "")
    if truncation_side:
        tokenizer.truncation_side = truncation_side

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [example_to_model_text(example, tokenizer, args) for example in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_len,
            padding=True,
            return_tensors="pt",
            # Chat templates already include model-specific special tokens.
            # Plain-text ablations still use normal tokenizer special tokens.
            add_special_tokens=args.use_chat_template != 1,
        )
        return tokenized

    return collate_fn
