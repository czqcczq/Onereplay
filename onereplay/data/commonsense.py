"""Commonsense170k loading, split, and tokenization for SFT / OPD."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk

from onereplay.data.chat import build_sft_tokenize_fn


def load_and_prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load Commonsense170k, split train/validation, and add tokenized fields."""

    dataset_dict = load_from_disk(args.dataset_path)
    dataset = dataset_dict["train"] if "train" in dataset_dict else dataset_dict
    split = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed, shuffle=True)
    train_dataset = split["train"]
    valid_dataset = split["test"]

    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples > 0:
        valid_dataset = valid_dataset.select(range(min(args.max_val_samples, len(valid_dataset))))

    add_tokenized_fields = build_sft_tokenize_fn(tokenizer, args.max_len)

    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        # In managed/sandboxed runs the dataset directory can be read-only.
        # Direct HuggingFace map cache files to an explicit writable location.
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = train_dataset.map(
            add_tokenized_fields,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "commonsense_train_tokenized.arrow"),
        )
        valid_dataset = valid_dataset.map(
            add_tokenized_fields,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "commonsense_valid_tokenized.arrow"),
        )
    else:
        train_dataset = train_dataset.map(add_tokenized_fields)
        valid_dataset = valid_dataset.map(add_tokenized_fields)
    return train_dataset, valid_dataset
