"""Scheduling checks for BatchMixedReplayLoader. No GPU or model needed.

Run before submitting a batch-level replay sweep:

    python -m onereplay.scripts.verify_batch_mix

It asserts the properties the sweep depends on: every micro-batch has the exact
new/replay composition, one epoch serves each new-task row exactly once, the
replay pool is cycled evenly rather than resampled with bias, epochs reshuffle,
and a given seed reproduces a given epoch.

The second half covers multi-domain mixing, where the property that matters is
that the domain ratio is spent exactly rather than sampled: an 0.8/0.2 arm and a
0.5/0.5 arm are supposed to differ in the ratio alone, so binomial noise on it
would blur the only variable. It also asserts that adding the machinery left the
single-pool schedule byte-identical, since the IF-line results were produced by
that path.
"""

from collections import Counter

from datasets import Dataset

from onereplay.data.batch_mix import BatchMixedReplayLoader, ReplayPool

NEW_TAG = 1000
REPLAY_TAG = 2000
MATH_TAG = 3000


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


def make_mixed_loader(new_dataset, pools, new_per_batch, replay_per_batch, seed=1):
    return BatchMixedReplayLoader(
        new_dataset=new_dataset,
        replay_pools=pools,
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


def check_mixing() -> None:
    """Weighted multi-pool draws: exact ratio, per-pool cycling, reproducibility."""

    new_dataset = make_dataset(100, NEW_TAG, tokens=3)
    if_pool = make_dataset(7, REPLAY_TAG, tokens=1)
    math_pool = make_dataset(11, MATH_TAG, tokens=4)

    def domains(loader):
        counts = Counter()
        for _, replay_ids in collect(loader):
            for row in replay_ids:
                counts["math" if row >= MATH_TAG else "if"] += 1
        return counts

    # 0.5/0.5 at 4 replay rows per batch must be exactly 2 + 2 every step.
    even = make_mixed_loader(
        new_dataset,
        [ReplayPool("if", if_pool, 0.5), ReplayPool("math", math_pool, 0.5)],
        new_per_batch=4,
        replay_per_batch=4,
    )
    print(even.describe())
    for _, replay_ids in collect(even):
        per_step = Counter("math" if row >= MATH_TAG else "if" for row in replay_ids)
        assert per_step["if"] == 2 and per_step["math"] == 2, per_step

    # 0.8/0.2 cannot be integral per step, so the guarantee is over the epoch:
    # 25 steps x 4 rows = 100 draws, 80 IF and 20 math, no drift.
    skewed = make_mixed_loader(
        new_dataset,
        [ReplayPool("if", if_pool, 0.8), ReplayPool("math", math_pool, 0.2)],
        new_per_batch=4,
        replay_per_batch=4,
    )
    counts = domains(skewed)
    assert counts["if"] == 80, counts
    assert counts["math"] == 20, counts
    print(f"0.8/0.2 at 4 replay rows/batch, one epoch: {dict(counts)} (exact, not sampled)")

    # The production config draws 8 replay rows per micro-batch, where 0.8 x 8 =
    # 6.4 is again fractional. Whatever the slot count, the epoch total has to
    # land on the asked-for share exactly.
    for replay_per_batch in (2, 4, 8, 16):
        arm = make_mixed_loader(
            new_dataset,
            [ReplayPool("if", if_pool, 0.8), ReplayPool("math", math_pool, 0.2)],
            new_per_batch=8,
            replay_per_batch=replay_per_batch,
        )
        counts = domains(arm)
        total = arm.replay_samples_per_epoch
        assert counts["if"] == round(0.8 * total), (replay_per_batch, counts, total)
        assert counts["math"] == round(0.2 * total), (replay_per_batch, counts, total)
        print(
            f"  replay_per_batch={replay_per_batch:>2}: "
            f"{counts['if']}/{counts['math']} of {total} = "
            f"{counts['if'] / total:.3f}/{counts['math'] / total:.3f}"
        )

    # Weights are normalized, so 4:1 is the same schedule as 0.8:0.2.
    unnormalized = make_mixed_loader(
        new_dataset,
        [ReplayPool("if", if_pool, 4.0), ReplayPool("math", math_pool, 1.0)],
        new_per_batch=4,
        replay_per_batch=4,
    )
    assert collect(unnormalized) == collect(
        make_mixed_loader(
            new_dataset,
            [ReplayPool("if", if_pool, 0.8), ReplayPool("math", math_pool, 0.2)],
            new_per_batch=4,
            replay_per_batch=4,
        )
    ), "weights are not normalized"

    # Each pool cycles on its own permutation, so coverage is even within a pool
    # even though the two pools are drawn at different rates.
    wide = make_mixed_loader(
        new_dataset,
        [ReplayPool("if", if_pool, 0.5), ReplayPool("math", math_pool, 0.5)],
        new_per_batch=4,
        replay_per_batch=4,
    )
    usage = Counter(row for _, replay_ids in collect(wide) for row in replay_ids)
    if_usage = [count for row, count in usage.items() if row < MATH_TAG]
    math_usage = [count for row, count in usage.items() if row >= MATH_TAG]
    assert len(if_usage) == 7 and len(math_usage) == 11, usage
    assert max(if_usage) - min(if_usage) <= 1, if_usage
    assert max(math_usage) - min(math_usage) <= 1, math_usage
    print(f"per-pool usage spread: if={sorted(if_usage)} math={sorted(math_usage)}")

    # A single pool passed through the mixing path must schedule exactly like
    # the legacy replay_dataset path, or the IF-line runs stop reproducing.
    legacy = make_loader(new_dataset, if_pool, new_per_batch=4, replay_per_batch=2)
    through_pools = make_mixed_loader(
        new_dataset, [ReplayPool("", if_pool, 1.0)], new_per_batch=4, replay_per_batch=2
    )
    assert collect(legacy) == collect(through_pools), "single-pool schedule changed"
    print("single-pool schedule unchanged by the mixing path")

    draws = even.draw_counts()
    assert set(draws) == {"if", "math"}, draws
    print(f"draw_counts after one epoch: {draws}")
    print("all multi-pool mixing checks passed")


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

    print("all batch_mix scheduling checks passed\n")
    check_mixing()


if __name__ == "__main__":
    main()
