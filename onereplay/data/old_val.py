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

``--old_val_presliced 1`` takes the other route: the caller has already written
the two files, one holding the rows C, F and replay consume and one holding the
held-out rows, and this module just reads the second. That is the only option
when "the rows the subset stops before" does not exist -- a corpus whose own
preparation drew its pool with ``rng.sample`` or a stratified quota has no
offset to point at, and ``load_self_distilled_pool`` ignores ``pool_size``
anyway, so a cut expressed only as a flag would leave the replay arm training
on the rows being scored. Splitting on disk makes that a line count rather than
an argument about whether two shuffles agree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from onereplay.data.chat import build_loader, build_sft_tokenize_fn
from onereplay.data.replay import cut_pool


def old_val_max_len(args: argparse.Namespace) -> int:
    """Token budget for the held-out rows; 0 falls back to --max_len.

    Sized per corpus for the same reason replay_max_len is (see data/replay.py):
    truncation keeps the *last* max_len tokens, so a row that does not fit loses
    its question and keeps its answer, and the number stops being a loss on
    question-answering. It has to match the budget the replay arm tokenizes that
    same corpus at, or the arms are scored on rows shaped differently from the
    ones they trained on -- FLAN at 512 while a DialogSum new task stays at 1024.
    """

    value = int(getattr(args, "old_val_max_len", 0) or 0)
    return value if value > 0 else args.max_len


def _slice_dataset(
    corpus_path: str,
    args: argparse.Namespace,
    tokenizer,
    cache_name: str,
    label: str,
):
    """One domain's held-out slice, cut from behind the old-knowledge subset.

    Every domain is cut at the same two offsets of its own shuffle, so a domain
    carried through several stages is scored on the same rows each time and its
    numbers subtract across stages.

    Returns the tokenized dataset rather than a loader so data/probe.py can put
    the same rows on a step-level schedule. The epoch-level loader gives one
    point per epoch, which on a two-epoch run is a before and an after, not a
    curve.
    """

    presliced = int(getattr(args, "old_val_presliced", 0)) == 1
    pool_size = int(getattr(args, "old_val_pool_size", 0))
    sample_seed = int(getattr(args, "old_val_sample_seed", -1))
    if not presliced and (pool_size <= 0 or sample_seed < 0):
        raise ValueError(
            "scoring a held-out domain needs --old_val_pool_size and "
            "--old_val_sample_seed, set to what collect_cov got as "
            "--max_samples/--sample_seed and the replay arm gets as "
            "--replay_pool_size/--replay_sample_seed. The slice is defined as the "
            "rows that subset stops before, so a wrong value here hands the replay "
            "arm rows it trains on. Pass --old_val_presliced 1 instead when the "
            "file already holds only held-out rows."
        )
    rows = int(getattr(args, "old_val_rows", -1))
    if rows < 0:
        rows = args.max_val_samples
    if rows <= 0:
        raise ValueError(f"{label} is set but --old_val_rows resolved to 0")

    corpus = load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        cache_dir=str(getattr(args, "replay_cache_dir", "") or "") or None,
    )
    if presliced:
        # The caller already split the corpus: this file holds the held-out rows
        # and another file holds the ones C, F and replay consume. Reproducing a
        # shuffle here would be reasoning about a cut that has already happened
        # on disk, where it can be counted instead. Used when the pool cut cannot
        # be expressed as "shuffle(seed) then a prefix" -- a corpus whose own
        # preparation sampled it (rng.sample, stratified draws) has no "rows the
        # subset stops before" to point at.
        dataset = corpus if rows >= len(corpus) else corpus.select(range(rows))
        origin = f"{len(dataset)} pre-split rows of {corpus_path}"
    else:
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
            print(f"{label}: only {available} rows lie outside the subset, not {rows}")
            rows = available
        dataset = shuffled.select(range(pool_size, pool_size + rows))
        origin = (
            f"rows {pool_size}..{pool_size + rows} of {corpus_path} shuffled with "
            f"seed {sample_seed}, outside the {pool_size}-row subset"
        )

    # {question, response} is what the prepare_* scripts write, but a corpus
    # that reaches this by way of a replay pool keeps its own names -- FLAN is
    # {inputs, targets}. Which pair is right is a property of the file, and the
    # wrong one does not fail near the flag that caused it: it raises a KeyError
    # from inside a datasets worker.
    input_column = str(getattr(args, "old_val_input_column", "") or "question")
    target_column = str(getattr(args, "old_val_target_column", "") or "response")
    missing = [name for name in (input_column, target_column) if name not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{corpus_path} has no column(s) {missing}; available: {dataset.column_names}. "
            "Set --old_val_input_column / --old_val_target_column to the pair the replay "
            "arm reads from this same corpus."
        )
    dataset = dataset.map(
        lambda example: {
            "instruction": str(example[input_column] or "").strip(),
            "input": "",
            "output": str(example[target_column] or "").strip(),
        },
        remove_columns=dataset.column_names,
    )
    # An empty side means no supervised tokens, which would be a NaN in the mean.
    dataset = dataset.filter(lambda example: bool(example["instruction"]) and bool(example["output"]))

    max_len = old_val_max_len(args)
    tokenize = build_sft_tokenize_fn(tokenizer, max_len)
    map_cache_dir = getattr(args, "map_cache_dir", "")
    if map_cache_dir:
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Its own cache file: the new task's val split is mapped into
        # commonsense_valid_tokenized.arrow and is still open on it, and each
        # prior domain needs a file of its own for the same reason.
        dataset = dataset.map(
            tokenize,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / cache_name),
        )
    else:
        dataset = dataset.map(tokenize)

    budget = f"max_len={max_len}" + (
        f" (new task stays at {args.max_len})" if max_len != args.max_len else ""
    )
    print(
        f"{label}: {len(dataset)} rows from {origin}, "
        f"columns {input_column}/{target_column}, {budget}"
    )
    return dataset


def _slice_loader(
    corpus_path: str,
    args: argparse.Namespace,
    tokenizer,
    cache_name: str,
    label: str,
    dataset=None,
):
    """Epoch-level loader over one domain's held-out slice."""

    if dataset is None:
        dataset = _slice_dataset(corpus_path, args, tokenizer, cache_name, label)
    batch_size = args.eval_batch_size or args.batch_size
    print(f"{label}: epoch-level loader at batch size {batch_size}")
    return build_loader(dataset, tokenizer, batch_size=batch_size, train=False)


def build_old_val_dataset(args: argparse.Namespace, tokenizer):
    """The old domain's held-out slice, tokenized, or None when disabled.

    Same rows and same rendering as build_old_val_loader; the probes take this
    one so the step-level curve and the epoch-level number cannot drift apart.
    """

    corpus_path = str(getattr(args, "old_val_jsonl", "") or "")
    if not corpus_path:
        return None
    return _slice_dataset(corpus_path, args, tokenizer, "old_val_tokenized.arrow", "old val")


def build_old_val_loader(args: argparse.Namespace, tokenizer, dataset=None):
    """Loader over the old domain's held-out slice, or None when disabled.

    dataset is what build_old_val_dataset already returned. It has to be passed
    whenever both routes are alive in one process -- the epoch-level loader here
    and the step-level probe -- because both map into the same
    old_val_tokenized.arrow with load_from_cache_file=False. The second map
    renames a freshly written file over that path, so the first dataset is left
    memory-mapped onto an inode nothing links to any more, and the next page it
    has to fault in kills the process with SIGBUS instead of raising. That read
    is the end-of-epoch old_val_loss, i.e. after the last training step of the
    run, which is the most expensive place to lose a job.
    """

    corpus_path = str(getattr(args, "old_val_jsonl", "") or "")
    if not corpus_path:
        return None
    return _slice_loader(
        corpus_path, args, tokenizer, "old_val_tokenized.arrow", "old val", dataset=dataset
    )


def build_prior_val_loaders(args: argparse.Namespace, tokenizer):
    """Domains learned before the one this stage started from, label by label.

    A chain longer than two stages leaves domains that are neither the new task
    nor the checkpoint's immediate predecessor: at medical -> law, finance was
    learned two stages back and nothing would report on it. Its rows are cut the
    same way old_val cuts, so the number is comparable to the one the previous
    stage recorded as old_val_loss for the same domain.

    Returns an ordered list of (label, loader); empty when the flag is unset,
    which leaves every metrics record byte-identical to before.
    """

    spec = str(getattr(args, "prior_val_jsonl", "") or "")
    if not spec:
        return []

    loaders = []
    seen: set[str] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--prior_val_jsonl takes label=path entries, got {item!r}. The label "
                "becomes the metrics column old_val_loss_<label>, so it cannot be "
                "guessed from the filename without the column silently renaming "
                "itself when the corpus moves."
            )
        label, _, path = item.partition("=")
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError(f"--prior_val_jsonl entry {item!r} has an empty label or path")
        if label in seen:
            raise ValueError(
                f"--prior_val_jsonl repeats the label {label!r}; the second one would "
                "overwrite the first one's column"
            )
        seen.add(label)
        loaders.append(
            (
                label,
                _slice_loader(
                    path,
                    args,
                    tokenizer,
                    f"prior_val_{label}_tokenized.arrow",
                    f"prior val [{label}]",
                ),
            )
        )
    return loaders


__all__ = [
    "build_old_val_dataset",
    "build_old_val_loader",
    "build_prior_val_loaders",
    "old_val_max_len",
]
