"""
Download Hugging Face datasets to a local folder.

Default behavior downloads BoolQ:

    python3 code/process_dataset/download_dataset.py

This saves the SuperGLUE BoolQ DatasetDict to:

    /home/weiliu1/huggingface/datasets/boolq

You can override the destination:

    python3 code/process_dataset/download_dataset.py \
        --output-dir /home/weiliu1/huggingface/datasets/boolq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "google/boolq"
DEFAULT_DATASET_CONFIG = ""
DEFAULT_OUTPUT_DIR = Path("/home/weiliu1/huggingface/datasets/boolq")
DEFAULT_CACHE_DIR = Path("/home/weiliu1/huggingface/datasets/cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and save it with save_to_disk()."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help='Hugging Face dataset name. Default: "google/boolq".',
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="Dataset config/subset. Default: empty, because google/boolq has no config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the dataset. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Dataset download cache directory. Default: {DEFAULT_CACHE_DIR}",
    )
    return parser.parse_args()


def split_counts(dataset: Any) -> str:
    if not hasattr(dataset, "keys"):
        return f"{len(dataset)} rows"
    return ", ".join(f"{split}: {len(dataset[split])}" for split in dataset.keys())


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it first, for example: "
            "pip install datasets"
        ) from exc

    dataset_label = args.dataset_name
    if args.dataset_config:
        dataset_label = f"{args.dataset_name}/{args.dataset_config}"
    print(f"Downloading dataset: {dataset_label}")
    try:
        load_kwargs = {"cache_dir": str(args.cache_dir)}
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)
        else:
            dataset = load_dataset(args.dataset_name, **load_kwargs)
    except Exception:
        if args.dataset_name != "super_glue":
            raise
        fallback_name = "google/boolq"
        print(f"Retrying with canonical dataset name: {fallback_name}")
        dataset = load_dataset(fallback_name, cache_dir=str(args.cache_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    print(f"Saved dataset to: {args.output_dir}")
    print(f"Splits: {split_counts(dataset)}")


if __name__ == "__main__":
    main()
