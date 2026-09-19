"""A fixed slice of the old domain, scored next to the new task's val split.

``val_loss`` says whether this stage learned the new task. What happened to the
domain the model arrived with has been left entirely to the benchmarks, and
those move whenever generation or parsing moves -- a model that starts phrasing
answers differently reads as a collapse to whichever parser stops recognizing
it. A cross-entropy on fixed old rows moves only with the weights.

Which rows: the ones the old-knowledge subset ends before.

The natural choice is the validation split that domain's own SFT run used, cut
from the pool with ``train_test_split(test_size=f, seed=S)``. It cannot be used
here. That call permutes the pool and takes the *leading* fraction of the
permutation, while the old-knowledge subset is ``shuffle(seed=S)`` plus the
first ``pool_size`` rows -- the same permutation, a longer prefix. With both
seeds at their default 42 the validation split is a strict subset of the pool:
all 340 rows of a 34k-row medical pool sit inside the 10k that produce C, F and
the replay arm's rehearsal corpus. The replay arm cycles that corpus several
times per run, so it would be scored on rows it had just spent epochs training
on, and its retention number would be part memorization.

So the slice is rows ``[pool_size, pool_size + n)`` of the same shuffle: the
rows the subset stops before. No arm trains on them during this stage, which is
the property the four-way comparison needs. Stage 1 did train on them (it
trained on 99% of the corpus), so the baseline is a fitted-data loss and sits
below what that run recorded as val_loss -- which does not matter, because what
is read here is the change from that baseline on rows held fixed across arms.
There is no slice that is outside both the pool and stage 1's training data:
stage 1's only untrained rows are the ones the pool swallowed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from onereplay.data.chat import build_loader, build_sft_tokenize_fn
from onereplay.data.replay import cut_pool


def build_old_val_loader(args: argparse.Namespace, tokenizer):
    """Loader over the old domain's held-out slice, or None when disabled."""

    corpus_path = str(getattr(args, "old_val_jsonl", "") or "")
    if not corpus_path:
        return None

    pool_size = int(getattr(args, "old_val_pool_size", 0))
    sample_seed = int(getattr(args, "old_val_sample_seed", -1))
    if pool_size <= 0 or sample_seed < 0:
        raise ValueError(
            "--old_val_jsonl needs --old_val_pool_size and --old_val_sample_seed, "
            "set to what collect_cov got as --max_samples/--sample_seed and the "
            "replay arm gets as --replay_pool_size/--replay_sample_seed. The "
            "slice is defined as the rows that subset stops before, so a wrong "
            "value here hands the replay arm rows it trains on."
        )
    rows = int(getattr(args, "old_val_rows", -1))
    if rows < 0:
        rows = args.max_val_samples
    if rows <= 0:
        raise ValueError("--old_val_jsonl is set but --old_val_rows resolved to 0")

    corpus = load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        cache_dir=str(getattr(args, "replay_cache_dir", "") or "") or None,
    )
    # pool_size 0 keeps every row: one permutation, cut at two offsets.
    shuffled = cut_pool(corpus, sample_seed, 0)
    available = len(shuffled) - pool_size
    if available <= 0:
        raise ValueError(
            f"{corpus_path} holds {len(shuffled)} rows and the old-knowledge subset "
            f"takes the first {pool_size}, so no row is left that the replay arm "
            "does not train on."
        )
    if available < rows:
        print(f"old val: only {available} rows lie outside the subset, not {rows}")
        rows = available

    dataset = shuffled.select(range(pool_size, pool_size + rows))
    dataset = dataset.map(
        lambda example: {
            "instruction": str(example["question"] or "").strip(),
            "input": "",
            "output": str(example["response"] or "").strip(),
        },
        remove_columns=dataset.column_names,
    )
    # An empty side means no supervised tokens, which would be a NaN in the mean.
    dataset = dataset.filter(lambda example: bool(example["instruction"]) and bool(example["output"]))

    tokenize = build_sft_tokenize_fn(tokenizer, args.max_len)
    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Its own cache file: the new task's val split is mapped into
        # commonsense_valid_tokenized.arrow and is still open on it.
        dataset = dataset.map(
            tokenize,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / "old_val_tokenized.arrow"),
        )
    else:
        dataset = dataset.map(tokenize)

    batch_size = args.eval_batch_size or args.batch_size
    print(
        f"old val: {len(dataset)} rows {pool_size}..{pool_size + rows} of {corpus_path} "
        f"shuffled with seed {sample_seed}, outside the {pool_size}-row subset, "
        f"batch size {batch_size}"
    )
    return build_loader(dataset, tokenizer, batch_size=batch_size, train=False)


__all__ = ["build_old_val_loader"]
