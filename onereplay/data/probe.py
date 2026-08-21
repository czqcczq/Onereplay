"""Fixed probe sets scored periodically during training.

The epoch-level val_loss only says how well the new task is being learned. It
says nothing about what happens to the old knowledge *while* training runs, and
three points per run cannot show the shape of anything. These probes add a
step-level view on the side the replay baseline is supposed to protect.

Three sets, all scored with the same forward pass the trainer uses:

  flan_heldout  FLAN rows the training pool never contained, self-distilled the
                same way. Measures whether the old *ability* survives.
  flan_inpool   Rows drawn from the pool replay actually trains on. Measures
                how far the model has memorized those specific rows.
  cs_val        The existing Commonsense validation split, so the new task is
                on the same axis.

Reading heldout alone is ambiguous: a rising curve could mean the model is
overfitting the replay pool or simply that the new task is dragging everything.
The gap between heldout and inpool separates the two, and it only means
something because both sides are scored on identical targets (the base model's
own answers) through identical tokenization.

A vanilla run should show the two FLAN curves lying on top of each other, since
neither slice is in its training data. That coincidence is the control: any gap
a replay run opens is caused by replay and nothing else.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from onereplay.data.chat import build_sft_tokenize_fn
from onereplay.data.replay import load_self_distilled_pool, to_sft_schema

TOKEN_COLUMNS = ["input_ids", "labels", "attention_mask"]


def _loader_namespace(args: argparse.Namespace, self_distill_file: str) -> Namespace:
    """Adapt the probe flags onto the field names replay.py reads."""

    return Namespace(
        replay_self_distill_file=self_distill_file,
        replay_cache_dir=getattr(args, "replay_cache_dir", ""),
        replay_drop_truncated=getattr(args, "replay_drop_truncated", 1),
        replay_input_column=getattr(args, "replay_input_column", "inputs"),
        replay_target_column=getattr(args, "replay_target_column", "targets"),
    )


def _subsample(dataset, size: int):
    """Take `size` rows spread evenly across the dataset.

    Even striding rather than a random draw: the pool was already shuffled once
    with sample_seed, so position carries no information, and a stride needs no
    seed of its own to be reproducible across runs.
    """

    if size <= 0 or size >= len(dataset):
        return dataset
    stride = len(dataset) // size
    return dataset.select(range(0, stride * size, stride))


def _tokenize(dataset, tokenizer, max_len: int, map_cache_dir: str, cache_name: str):
    tokenize = build_sft_tokenize_fn(tokenizer, max_len)
    if map_cache_dir:
        cache_dir = Path(map_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return dataset.map(
            tokenize,
            load_from_cache_file=False,
            cache_file_name=str(cache_dir / cache_name),
        )
    return dataset.map(tokenize)


def build_probe_loader(dataset, tokenizer, batch_size: int) -> DataLoader:
    """Wrap an already tokenized dataset in a fixed-order eval loader.

    shuffle=False is load-bearing twice over: the probe must score the same
    rows in the same batches every time or the curve moves for reasons that
    have nothing to do with the model, and a shuffling loader would draw from
    the global RNG, which would desynchronize the training run from a
    probe-free one and cost the ability to check the two against each other.
    """

    tokenizer.padding_side = "left"
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    extra = [column for column in dataset.column_names if column not in TOKEN_COLUMNS]
    if extra:
        dataset = dataset.remove_columns(extra)
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=False)


def build_self_distilled_probe(
    args: argparse.Namespace,
    tokenizer,
    self_distill_file: str,
    size: int,
    cache_name: str,
):
    """Load, cut and tokenize one self-distilled FLAN slice.

    Both FLAN probes come through here so the two curves differ only in which
    rows they hold, never in how those rows were rendered.
    """

    loader_args = _loader_namespace(args, self_distill_file)
    pool = load_self_distilled_pool(loader_args)
    dataset = to_sft_schema(pool, loader_args)
    dataset = _subsample(dataset, size)
    return _tokenize(
        dataset,
        tokenizer,
        args.max_len,
        getattr(args, "map_cache_dir", ""),
        cache_name,
    )


def build_probe_loaders(args: argparse.Namespace, tokenizer, valid_dataset) -> dict[str, DataLoader]:
    """Assemble every probe the flags ask for, keyed by curve name.

    Returns an empty dict when --probe_every is 0, which is the switch that
    keeps this whole path out of the production runs whose timings are
    reported.
    """

    if args.probe_every <= 0:
        return {}

    batch_size = args.probe_batch_size or args.eval_batch_size or args.batch_size
    loaders: dict[str, DataLoader] = {}

    if args.probe_heldout_file:
        heldout = build_self_distilled_probe(
            args,
            tokenizer,
            args.probe_heldout_file,
            args.probe_heldout_size,
            "probe_flan_heldout_tokenized.arrow",
        )
        loaders["flan_heldout"] = build_probe_loader(heldout, tokenizer, batch_size)
        print(f"probe flan_heldout: {len(heldout)} rows from {args.probe_heldout_file}")

    inpool_file = args.probe_inpool_file or args.replay_self_distill_file
    if inpool_file:
        inpool = build_self_distilled_probe(
            args,
            tokenizer,
            inpool_file,
            args.probe_inpool_size,
            "probe_flan_inpool_tokenized.arrow",
        )
        loaders["flan_inpool"] = build_probe_loader(inpool, tokenizer, batch_size)
        print(f"probe flan_inpool: {len(inpool)} rows from {inpool_file}")

    if args.probe_cs_val_size != 0 and valid_dataset is not None:
        subset = _subsample(valid_dataset, args.probe_cs_val_size)
        loaders["cs_val"] = build_probe_loader(subset, tokenizer, batch_size)
        print(f"probe cs_val: {len(subset)} rows")

    if not loaders:
        raise ValueError(
            "--probe_every is set but no probe set was built; pass "
            "--probe_heldout_file and/or --probe_inpool_file"
        )
    return loaders


__all__ = ["build_probe_loader", "build_probe_loaders", "build_self_distilled_probe"]
