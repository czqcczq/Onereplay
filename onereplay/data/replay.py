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

The self-distilled file keeps the corpus's own answer in gold_targets on every
row, so --replay_self_distill_file FILE --replay_target_column gold_targets is
a third configuration: gold answers on the self-distilled file's exact rows, in
its exact order. That is the ablation to run when the suspicion is that the
base model's own answers are simply wrong often enough to poison the anchor --
pointing --replay_data_files back at the raw corpus would re-roll the shuffle
and the pool cut at the same time, and could not tell the two causes apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk

from onereplay.data.batch_mix import ReplayPool
from onereplay.data.chat import build_sft_tokenize_fn


def replay_max_len(args: argparse.Namespace) -> int:
    """Token budget for replay rows, independent of the new task's --max_len.

    The new task fixes --max_len across every arm, so vanilla, OneReplay and
    replay tokenize Commonsense170k identically; raising it for one arm would
    add a second variable. Old-knowledge rows are a different matter, because
    each corpus has its own length: FLAN self-distilled answers top out at 503
    tokens together with their prompt, while MetaMath ones reach 2008. Training
    truncation keeps the last max_len tokens, so a math row that does not fit
    keeps its answer and loses its question, which turns a question-answer
    rehearsal into bare continuation -- at 512 that happens to 16.2% of the
    MetaMath pool, at 2048 to none of it.

    Sizing this per corpus is also what keeps the comparison against OneReplay
    honest: C_math was collected at max_len 2048, so a replay arm capped at 512
    would be protecting from strictly less old knowledge than the penalty
    encodes. 0 falls back to --max_len, which is what every pre-existing run
    used.
    """

    value = int(getattr(args, "replay_max_len", 0) or 0)
    return value if value > 0 else args.max_len


def replay_target_flavor(args: argparse.Namespace) -> str:
    """Name the answers replay is about to train on, for the run log.

    Follows the column rather than the filename: reading gold_targets out of a
    self-distilled file is the gold ablation, and a log that called it
    self-distilled would mislabel the one run whose whole point is the switch.

    --replay_mix_files carries those same self-distilled files, one per domain,
    so it has to count as a self-distilled source too. Checking only the
    single-file flag labels every multi-domain arm "gold-target" whatever
    column it actually trained on, which is exactly backwards for the arms
    that kept the default targets column.
    """

    column = getattr(args, "replay_target_column", "") or "targets"
    from_self_distilled_file = bool(
        getattr(args, "replay_self_distill_file", "") or getattr(args, "replay_mix_files", "")
    )
    if not from_self_distilled_file:
        return "gold-target"
    if column == "targets":
        return "self-distilled"
    return f"gold-target (column {column} of the self-distilled file)"


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


def load_self_distilled_pool(args: argparse.Namespace, self_distill_file: str = ""):
    """Load base-model answers written by scripts/generate_replay_targets.py.

    That script already applied the shuffle, the pool cut and the schema
    filtering, and stamped every row with its position in the pool. Sorting by
    that index restores the exact order the gold-target route produces, so the
    two flavors train on the same rows and only the targets differ. No second
    shuffle here: it would break the nesting.
    """

    dataset = load_dataset(
        "json",
        data_files=self_distill_file or args.replay_self_distill_file,
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


def build_replay_dataset(
    args: argparse.Namespace,
    tokenizer,
    num_samples: int | None = None,
    self_distill_file: str = "",
    label: str = "",
):
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

    if self_distill_file or getattr(args, "replay_self_distill_file", ""):
        pool = load_self_distilled_pool(args, self_distill_file)
    else:
        pool = load_replay_pool(args)
        pool = pool.shuffle(seed=args.replay_sample_seed)
        if args.replay_pool_size > 0:
            pool = pool.select(range(min(args.replay_pool_size, len(pool))))

    dataset = to_sft_schema(pool, args)
    tag = f"replay pool[{label}]" if label else "replay pool"
    if num_samples is None:
        print(f"{tag}: {len(dataset)} usable rows (no cut)")
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

    max_len = replay_max_len(args)
    if max_len != args.max_len:
        print(f"{tag}: tokenizing at max_len={max_len} (new task stays at {args.max_len})")
    tokenize = build_sft_tokenize_fn(tokenizer, max_len)
    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # One cache file per pool: a shared name would make the second pool read
        # back the first pool's tokenized rows.
        suffix = f"_{label}" if label else ""
        return dataset.map(
            tokenize,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / f"replay_train_tokenized{suffix}.arrow"),
        )
    return dataset.map(tokenize)


def parse_replay_mix(args: argparse.Namespace) -> list[tuple[str, str, float]]:
    """Parse --replay_mix_files / --replay_mix_weights into (label, path, weight).

    Spelled the same way mix_covariances.py takes its inputs, so the OneReplay
    side and the replay side of a multi-domain experiment are configured in
    parallel: C_mix gets --inputs/--weights over covariance files, the replay
    arm gets --replay_mix_files/--replay_mix_weights over corpora. Weights are
    row shares and are normalized, so "0.8,0.2" and "4,1" mean the same thing.
    """

    spec = getattr(args, "replay_mix_files", "") or ""
    entries: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--replay_mix_files expects LABEL=PATH entries, got {item!r}. The label goes "
                "into the run log so the token shares can be read per domain."
            )
        label, path = item.split("=", 1)
        entries.append((label.strip(), path.strip()))
    if not entries:
        return []

    raw = getattr(args, "replay_mix_weights", "") or ""
    weights = [float(piece) for piece in raw.split(",") if piece.strip()]
    if not weights:
        weights = [1.0] * len(entries)
    if len(weights) != len(entries):
        raise ValueError(
            f"--replay_mix_weights has {len(weights)} values for {len(entries)} pools; "
            "give one row share per pool or leave it empty for equal shares"
        )
    if any(weight <= 0 for weight in weights):
        raise ValueError("--replay_mix_weights must all be > 0")
    total = sum(weights)
    return [(label, path, weight / total) for (label, path), weight in zip(entries, weights)]


def build_replay_pools(args: argparse.Namespace, tokenizer) -> list[ReplayPool]:
    """Build every replay pool the flags ask for, tokenized and weighted.

    One pool for the single-corpus arms, several for a multi-domain arm. Each
    pool keeps all of its usable rows; the ratio lives in the weights and is
    spent by the loader's scheduler. Encoding the ratio by truncating a corpus
    instead would change how often its rows repeat, which is a second variable.
    """

    print(f"replay targets: {replay_target_flavor(args)}")
    mix = parse_replay_mix(args)
    if not mix:
        return [ReplayPool(label="", dataset=build_replay_dataset(args, tokenizer), weight=1.0)]

    pools: list[ReplayPool] = []
    for label, path, weight in mix:
        dataset = build_replay_dataset(args, tokenizer, self_distill_file=path, label=label)
        pools.append(ReplayPool(label=label, dataset=dataset, weight=weight))
    print(
        "replay mix: "
        + ", ".join(
            f"{pool.label}={len(pool.dataset)} rows @ row share {pool.weight:.4f}"
            for pool in pools
        )
    )
    return pools


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
    flavor = replay_target_flavor(args)
    print(
        f"{flavor} replay: {len(train_dataset)} new-task + {len(replay_dataset)} replay "
        f"= {len(mixed)} rows (ratio={args.replay_ratio}, "
        f"actual={len(replay_dataset) / max(len(train_dataset), 1):.4f})"
    )
    return mixed
