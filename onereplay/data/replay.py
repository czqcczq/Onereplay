"""Experience replay: mix real old-knowledge samples into SFT training.

This is the head-to-head baseline for OneReplay. OneReplay compresses the
old-knowledge corpus into per-layer second moments C = E[x x^T] once and never
touches the corpus again; replay keeps the corpus and re-trains on a fraction
of it alongside the new task.

Both routes read the same FLAN dump with the same sampling seed, and the
replay subset is nested inside the pool that produced C, so the comparison
isolates *how* the old knowledge is used rather than how much of it is seen.

Two flavors of target, selected by --replay_self_distill_file:

  gold (default)  FLAN's own targets. These are bare one-line answers, which
                  for an instruction-tuned base model is a new task rather
                  than a rehearsal: it starts near loss 3.0, takes over most
                  of the gradient, and pulls the model toward FLAN's
                  answer-key style.
  self-distilled  The base model's own answers to the same prompts, produced
                  offline by scripts/generate_replay_targets.py. Replay then
                  starts near zero loss and only anchors the model to W0,
                  which is the same thing the OneReplay penalty approximates.
                  This is the information-matched comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk

from onereplay.data.chat import build_sft_tokenize_fn


def load_replay_pool(args: argparse.Namespace):
    """Load the raw old-knowledge corpus from save_to_disk or json/jsonl files."""

    if args.replay_dataset_path:
        dataset = load_from_disk(args.replay_dataset_path)
        return dataset[args.replay_split] if args.replay_split in dataset else dataset

    if not args.replay_data_files:
        raise ValueError(
            "replay_ratio > 0 requires --replay_data_files or --replay_dataset_path"
        )
    return load_dataset(
        "json",
        data_files=args.replay_data_files,
        split=args.replay_split,
        cache_dir=args.replay_cache_dir or None,
    )


def load_self_distilled_pool(args: argparse.Namespace):
    """Load base-model answers written by scripts/generate_replay_targets.py.

    That script already applied the shuffle, the pool cut and the schema
    filtering, and stamped every row with its position in the pool. Sorting by
    that index restores the exact order the gold-target route produces, so the
    two flavors train on the same rows and only the targets differ. No second
    shuffle here: it would break the nesting.
    """

    dataset = load_dataset(
        "json",
        data_files=args.replay_self_distill_file,
        split="train",
        cache_dir=args.replay_cache_dir or None,
    )
    if "index" in dataset.column_names:
        dataset = dataset.sort("index")

    drop_truncated = getattr(args, "replay_drop_truncated", 1) == 1
    if drop_truncated and "truncated" in dataset.column_names:
        before = len(dataset)
        # A cut-off answer has no stop token. Training on it would teach the
        # model not to end its turn, which is the failure the gold-target
        # baseline already suffers from.
        dataset = dataset.filter(lambda example: not example["truncated"])
        if before != len(dataset):
            print(f"self-distilled replay: dropped {before - len(dataset)} truncated rows")
    return dataset


def to_sft_schema(dataset, args: argparse.Namespace):
    """Rename the corpus onto the {instruction, input, output} training schema.

    collect_cov renders FLAN through the chat template with `inputs` as the
    user turn and `targets` as the assistant turn. Replay reuses that mapping
    so both routes see the same old knowledge in the same format.
    """

    input_column = args.replay_input_column
    target_column = args.replay_target_column
    missing = [
        column for column in (input_column, target_column) if column not in dataset.column_names
    ]
    if missing:
        raise ValueError(
            f"replay corpus is missing column(s) {missing}; available: {dataset.column_names}"
        )

    def convert(example):
        return {
            "instruction": str(example[input_column] or "").strip(),
            "input": "",
            "output": str(example[target_column] or "").strip(),
        }

    dataset = dataset.map(convert, remove_columns=dataset.column_names)
    # A row with an empty target has no supervised tokens, so every label would
    # be -100 and a batch of such rows would produce a NaN loss.
    return dataset.filter(lambda example: bool(example["instruction"]) and bool(example["output"]))


def build_replay_dataset(args: argparse.Namespace, tokenizer, num_samples: int | None = None):
    """Sample and tokenize old-knowledge rows.

    On the gold-target path the pool is shuffled with --replay_sample_seed and
    truncated to --replay_pool_size before anything else, mirroring what
    collect_cov did. Taking the first num_samples rows of that identically
    ordered pool keeps every replay subset nested inside the covariance pool.

    The self-distilled file already went through those same steps offline, so
    it is only sorted back into pool order and then cut the same way.

    num_samples=None returns the whole pool without cutting it. Batch-level
    mixing needs that: it draws a fixed number of replay rows per micro-batch
    and cycles through the pool as many times as the epoch requires, so the
    subset size is decided by the schedule rather than up front.
    """

    if getattr(args, "replay_self_distill_file", ""):
        pool = load_self_distilled_pool(args)
    else:
        pool = load_replay_pool(args)
        pool = pool.shuffle(seed=args.replay_sample_seed)
        if args.replay_pool_size > 0:
            pool = pool.select(range(min(args.replay_pool_size, len(pool))))

    dataset = to_sft_schema(pool, args)
    if num_samples is None:
        print(f"replay pool: {len(dataset)} usable rows (no cut)")
    elif num_samples > len(dataset):
        # Data-level mixing cannot honour a ratio larger than pool/train, and
        # silently training on fewer rows than asked would mislabel the run.
        raise ValueError(
            f"requested {num_samples} replay samples but the pool only holds {len(dataset)} "
            f"usable rows (actual ratio would be {len(dataset) / num_samples * args.replay_ratio:.4f} "
            f"instead of {args.replay_ratio}). Raise --replay_pool_size and regenerate the "
            "self-distilled corpus, or use --replay_per_batch for batch-level mixing, which "
            "cycles the pool instead of needing it to be large enough."
        )
    else:
        dataset = dataset.select(range(num_samples))

    tokenize = build_sft_tokenize_fn(tokenizer, args.max_len)
    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return dataset.map(
            tokenize,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "replay_train_tokenized.arrow"),
        )
    return dataset.map(tokenize)


def mix_replay_into_train(args: argparse.Namespace, tokenizer, train_dataset):
    """Append replay_ratio * |train| old-knowledge rows to the training set.

    Appending rather than substituting keeps the new-task sample count
    identical to the vanilla and OneReplay runs, so retention gains cannot be
    explained by having learned less of the new task. The price is a
    (1 + replay_ratio) times longer epoch, which is exactly the cost this
    baseline is meant to expose.
    """

    num_replay = int(round(args.replay_ratio * len(train_dataset)))
    if num_replay <= 0:
        print(f"replay_ratio={args.replay_ratio} rounds to 0 samples; training stays vanilla")
        return train_dataset

    replay_dataset = build_replay_dataset(args, tokenizer, num_replay)
    shared_columns = [
        column for column in train_dataset.column_names if column in replay_dataset.column_names
    ]
    train_dataset = train_dataset.remove_columns(
        [column for column in train_dataset.column_names if column not in shared_columns]
    )
    replay_dataset = replay_dataset.remove_columns(
        [column for column in replay_dataset.column_names if column not in shared_columns]
    )

    mixed = concatenate_datasets([train_dataset, replay_dataset]).shuffle(seed=args.seed)
    flavor = "self-distilled" if getattr(args, "replay_self_distill_file", "") else "gold-target"
    print(
        f"{flavor} replay: {len(train_dataset)} new-task + {len(replay_dataset)} replay "
        f"= {len(mixed)} rows (ratio={args.replay_ratio}, "
        f"actual={len(replay_dataset) / max(len(train_dataset), 1):.4f})"
    )
    return mixed
