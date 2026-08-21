"""Scheduling checks for BatchMixedReplayLoader. No GPU or model needed.

Run before submitting a batch-level replay sweep:

    python -m onereplay.scripts.verify_batch_mix

It asserts the properties the sweep depends on: every micro-batch has the exact
new/replay composition, one epoch serves each new-task row exactly once, the
replay pool is cycled evenly rather than resampled with bias, epochs reshuffle,
and a given seed reproduces a given epoch.
"""

from collections import Counter

from datasets import Dataset

from onereplay.data.batch_mix import BatchMixedReplayLoader

NEW_TAG = 1000
REPLAY_TAG = 2000


def make_dataset(size: int, tag: int, tokens: int):
    """Tag every row so a collated batch can be traced back to its source."""

    return Dataset.from_dict(
        {
            "input_ids": [[tag + i] for i in range(size)],
            "labels": [[tag + i] * tokens for i in range(size)],
            "attention_mask": [[1] for _ in range(size)],
        }
    )


def make_loader(new_dataset, replay_dataset, new_per_batch, replay_per_batch, seed=1):
    return BatchMixedReplayLoader(
        new_dataset=new_dataset,
        replay_dataset=replay_dataset,
        collate_fn=lambda rows: rows,
        new_per_batch=new_per_batch,
        replay_per_batch=replay_per_batch,
        seed=seed,
    )


def collect(loader):
    """Return per-step (new_ids, replay_ids) recovered from the tagged rows."""

    steps = []
    for batch in loader:
        new_ids = [row["input_ids"][0] for row in batch if row["input_ids"][0] < REPLAY_TAG]
        replay_ids = [row["input_ids"][0] for row in batch if row["input_ids"][0] >= REPLAY_TAG]
        steps.append((new_ids, replay_ids))
    return steps


def main() -> None:
    new_dataset = make_dataset(100, NEW_TAG, tokens=3)
    replay_dataset = make_dataset(7, REPLAY_TAG, tokens=1)

    loader = make_loader(new_dataset, replay_dataset, new_per_batch=4, replay_per_batch=2)
    assert len(loader) == 25, f"steps_per_epoch={len(loader)}, want 100//4=25"
    assert loader.new_samples_per_epoch == 100
    assert loader.replay_samples_per_epoch == 50
    print(loader.describe())

    epoch1 = collect(loader)
    assert len(epoch1) == 25, len(epoch1)
    for new_ids, replay_ids in epoch1:
        assert len(new_ids) == 4, new_ids
        assert len(replay_ids) == 2, replay_ids

    # New-task rows: exactly one pass, no repeats, full coverage. This is what
    # makes "new task volume per update" exact.
    seen_new = [row for new_ids, _ in epoch1 for row in new_ids]
    assert len(seen_new) == 100
    assert len(set(seen_new)) == 100, "a new-task row was served twice in one epoch"

    # Replay rows are cycled: 50 draws from a pool of 7 must hit every row 7 or
    # 8 times. A biased sampler would show a wider spread.
    replay_counts = Counter(row for _, replay_ids in epoch1 for row in replay_ids)
    assert sum(replay_counts.values()) == 50
    assert len(replay_counts) == 7, f"pool coverage {len(replay_counts)}/7"
    assert max(replay_counts.values()) - min(replay_counts.values()) <= 1, replay_counts
    print(f"replay usage per pool row: {sorted(replay_counts.values())}")

    # Epoch 2 reshuffles the new-task order and keeps full coverage.
    epoch2 = collect(loader)
    assert [new for new, _ in epoch1] != [new for new, _ in epoch2], "epoch did not reshuffle"
    assert len({row for new_ids, _ in epoch2 for row in new_ids}) == 100, "epoch 2 lost coverage"

    # Same seed reproduces the same epoch.
    replica = make_loader(new_dataset, replay_dataset, new_per_batch=4, replay_per_batch=2)
    assert collect(replica) == epoch1, "same seed did not reproduce the same epoch"

    # Tail handling: 100 rows at 6 per batch keeps 16 steps and drops 4.
    tail = make_loader(new_dataset, replay_dataset, new_per_batch=6, replay_per_batch=1)
    assert len(tail) == 16, len(tail)
    assert tail.new_samples_per_epoch == 96
    assert all(len(new) == 6 and len(replay) == 1 for new, replay in collect(tail))

    # An early break (max_steps) must still advance the epoch counter, otherwise
    # a truncated epoch would replay the identical order next time.
    partial = make_loader(new_dataset, replay_dataset, new_per_batch=4, replay_per_batch=2)
    iterator = iter(partial)
    next(iterator)
    assert partial.epoch == 1, partial.epoch

    print("all batch_mix scheduling checks passed")


if __name__ == "__main__":
    main()
