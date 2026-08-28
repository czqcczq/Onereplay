"""Strict batch-level replay mixing.

Data-level mixing (`replay.mix_replay_into_train`) appends replay rows to the
training set and shuffles globally, so the new/replay split inside a
micro-batch is binomial rather than fixed: with an equal number of replay and
new-task rows, an accumulation window of 16 micro-batches holds 64 +/- 5.7
new-task rows. This module composes every micro-batch explicitly instead --
`new_per_batch` new-task rows plus `replay_per_batch` replay rows -- so the
number of new-task rows behind every optimizer update is exact.

Callers hold the micro-batch size fixed at the baseline's value and carve the
replay rows out of it, so per-step time and peak memory stay directly
comparable with a vanilla run. The trainer's accum_steps = accumulation_size //
batch_size then counts micro-batches, not new-task rows, so accumulation_size
has to grow with the replay share to keep the new-task rows per update near the
baseline: at batch_size=8, replay_per_batch=4 needs accumulation_size=128 for
16 steps x 4 rows = 64. Shares whose new-task count does not divide the
baseline's 64 land a few percent off (7 rows x 9 steps = 63), which is far
below seed noise and leaves the per-epoch new-task total unchanged.

The replay pool is *cycled*, not sampled without replacement: at
replay_per_batch=4 one epoch consumes ~168k replay rows from a ~17k pool. That
is deliberate. The pool is exactly the set of rows that produced the covariance
C, so OneReplay and replay see the same old knowledge and differ only in how
they use it. Enlarging the pool to avoid repetition would instead hand replay
strictly more information than C encodes and break the comparison.

Several pools can be mixed at once, which is what a multi-domain arm needs: the
OneReplay side protects two domains through C_mix = w_if * C_if + w_math *
C_math, and the replay side's counterpart is drawing replay rows from both
corpora. Weights are *row* shares, and the scheduler spends them by
largest-remainder rather than by sampling, so an 0.8/0.2 split at
replay_per_batch=4 alternates 3 and 4 rows deterministically and hits the exact
ratio over every five steps instead of only in expectation. Keeping each pool
whole and weighting the draw is what lets the two arms differ in the ratio
alone: down-sampling a corpus to encode the ratio would also change how often
its rows repeat, and chapter 3 showed replay memorizes what it repeats.

Rows are not the unit the loss averages over, so describe() reports the
supervised-token share next to the row share. On the IF-only line a 50.00% row
share carried 93.16% of the supervised tokens, and a math pool whose answers
are four times longer than FLAN's moves that number again. Both shares are
printed at startup so a run can be checked against
scripts/stat_replay_pools.py before it burns GPU hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import DataCollatorForTokenClassification

TOKEN_COLUMNS = ["input_ids", "labels", "attention_mask"]


@dataclass
class ReplayPool:
    """One old-knowledge corpus and its target share of the replay rows."""

    label: str
    dataset: Any
    weight: float = 1.0


def _keep_token_columns(dataset):
    extra = [column for column in dataset.column_names if column not in TOKEN_COLUMNS]
    return dataset.remove_columns(extra) if extra else dataset


def _mean_supervised_tokens(dataset, limit: int = 512) -> float:
    """Average count of label positions that carry loss (labels != -100).

    A row's share of the batch is not its share of the loss: the model averages
    cross-entropy over supervised tokens, so a batch that is half replay *rows*
    is half replay *loss* only when both sides have similar answer lengths.
    """

    count = min(limit, len(dataset))
    if count == 0:
        return 0.0
    rows = dataset[list(range(count))]["labels"]
    return sum(sum(1 for token in row if token != -100) for row in rows) / count


def _normalize_pools(
    replay_dataset,
    replay_pools: Sequence[ReplayPool | tuple] | None,
) -> list[ReplayPool]:
    """Accept either a single dataset or a weighted list, and normalize weights."""

    if replay_pools is not None and replay_dataset is not None:
        raise ValueError("pass either replay_dataset or replay_pools, not both")

    if replay_pools is None:
        if replay_dataset is None:
            raise ValueError("batch-level replay needs replay_dataset or replay_pools")
        pools = [ReplayPool(label="", dataset=replay_dataset, weight=1.0)]
    else:
        pools = [
            pool if isinstance(pool, ReplayPool) else ReplayPool(*pool) for pool in replay_pools
        ]

    if not pools:
        raise ValueError("replay_pools is empty")
    for pool in pools:
        if len(pool.dataset) == 0:
            raise ValueError(f"replay pool {pool.label or '<unnamed>'} is empty")
        if pool.weight <= 0:
            raise ValueError(
                f"replay pool {pool.label or '<unnamed>'} has weight {pool.weight}; a pool that "
                "never gets drawn should be left out rather than weighted zero"
            )
    total = sum(pool.weight for pool in pools)
    return [ReplayPool(pool.label, pool.dataset, pool.weight / total) for pool in pools]


class BatchMixedReplayLoader:
    """Iterable of micro-batches with a fixed new/replay composition."""

    def __init__(
        self,
        new_dataset,
        collate_fn,
        new_per_batch: int,
        replay_per_batch: int,
        replay_dataset=None,
        replay_pools: Sequence[ReplayPool | tuple] | None = None,
        seed: int = 1,
    ) -> None:
        if new_per_batch <= 0:
            raise ValueError("new_per_batch must be >= 1")
        if replay_per_batch <= 0:
            raise ValueError("replay_per_batch must be >= 1; use build_loader for no replay")

        self.new_dataset = new_dataset
        self.pools = _normalize_pools(replay_dataset, replay_pools)
        self.collate_fn = collate_fn
        self.new_per_batch = new_per_batch
        self.replay_per_batch = replay_per_batch
        self.seed = int(seed)

        # Drop the tail that cannot fill a micro-batch, so every step carries
        # exactly new_per_batch new-task rows and the update boundary is exact.
        self.steps_per_epoch = len(new_dataset) // new_per_batch
        if self.steps_per_epoch == 0:
            raise ValueError(
                f"new dataset has {len(new_dataset)} rows, fewer than new_per_batch={new_per_batch}"
            )

        self.epoch = 0
        self._orders: list[list[int]] = [[] for _ in self.pools]
        self._cursors: list[int] = [0 for _ in self.pools]
        self._passes: list[int] = [0 for _ in self.pools]
        self._draws: list[int] = [0 for _ in self.pools]
        # Fractional row budget carried between steps. Spending it by
        # largest-remainder is what makes a non-integer share exact over a few
        # steps instead of only on average.
        self._quota_debt: list[float] = [0.0 for _ in self.pools]

    @property
    def replay_dataset(self):
        """The only pool, for the single-corpus callers that predate mixing."""

        if len(self.pools) != 1:
            raise AttributeError(
                f"this loader mixes {len(self.pools)} replay pools; use .pools instead"
            )
        return self.pools[0].dataset

    @property
    def new_samples_per_epoch(self) -> int:
        return self.steps_per_epoch * self.new_per_batch

    @property
    def replay_samples_per_epoch(self) -> int:
        return self.steps_per_epoch * self.replay_per_batch

    def describe(self) -> str:
        dropped = len(self.new_dataset) - self.new_samples_per_epoch
        new_tokens = _mean_supervised_tokens(self.new_dataset)
        pool_tokens = [_mean_supervised_tokens(pool.dataset) for pool in self.pools]

        replay_tokens_per_row = sum(
            pool.weight * tokens for pool, tokens in zip(self.pools, pool_tokens)
        )
        row_share = self.replay_per_batch / (self.new_per_batch + self.replay_per_batch)
        token_total = self.new_per_batch * new_tokens + self.replay_per_batch * replay_tokens_per_row
        token_share = (
            self.replay_per_batch * replay_tokens_per_row / token_total if token_total else 0.0
        )

        lines = [
            f"batch-level replay: {self.new_per_batch} new + {self.replay_per_batch} replay "
            f"per micro-batch (micro-batch={self.new_per_batch + self.replay_per_batch}), "
            f"{self.steps_per_epoch} steps/epoch, "
            f"{self.new_samples_per_epoch} new-task rows/epoch (dropped {dropped} tail rows), "
            f"{self.replay_samples_per_epoch} replay rows/epoch from {len(self.pools)} pool(s)"
        ]
        for pool, tokens in zip(self.pools, pool_tokens):
            rows_per_epoch = pool.weight * self.replay_samples_per_epoch
            within = pool.weight * tokens / replay_tokens_per_row if replay_tokens_per_row else 0.0
            lines.append(
                f"  pool[{pool.label or 'replay'}]: row weight={pool.weight:.4f} "
                f"({pool.weight * self.replay_per_batch:.2f} rows/batch), "
                f"{rows_per_epoch:.0f} rows/epoch from a pool of {len(pool.dataset)} "
                f"(cycled {rows_per_epoch / len(pool.dataset):.2f}x per epoch), "
                f"mean supervised tokens={tokens:.1f}, token share within replay={within:.4f}"
            )
        lines.append(
            f"  replay share of rows={row_share:.4f}, of supervised tokens={token_share:.4f} "
            f"(mean supervised tokens: new={new_tokens:.1f}, "
            f"replay={replay_tokens_per_row:.1f})"
        )
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _permutation(self, size: int, seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(size, generator=generator).tolist()

    def _next_counts(self) -> list[int]:
        """Split this step's replay slots across the pools by largest remainder.

        Deterministic on purpose: a stochastic draw would put binomial noise on
        the domain ratio, which is the one variable the multi-domain arms differ
        in. With one pool this returns [replay_per_batch] every step, so the
        single-corpus schedule is byte-identical to the pre-mixing version.
        """

        if len(self.pools) == 1:
            return [self.replay_per_batch]

        for index, pool in enumerate(self.pools):
            self._quota_debt[index] += pool.weight * self.replay_per_batch
        counts = [int(debt) for debt in self._quota_debt]
        remaining = self.replay_per_batch - sum(counts)
        if remaining > 0:
            order = sorted(
                range(len(self.pools)),
                key=lambda index: self._quota_debt[index] - counts[index],
                reverse=True,
            )
            for index in order[:remaining]:
                counts[index] += 1
        for index, count in enumerate(counts):
            self._quota_debt[index] -= count
        return counts

    def _next_replay_indices(self, pool_index: int, count: int) -> list[int]:
        """Take the next `count` indices from one pool, reshuffling at wrap."""

        pool_size = len(self.pools[pool_index].dataset)
        indices: list[int] = []
        while len(indices) < count:
            if self._cursors[pool_index] >= len(self._orders[pool_index]):
                # A fresh permutation per pass keeps the pool from pairing the
                # same replay rows with the same new-task rows every cycle. The
                # pool_index term leaves pool 0's stream unchanged, so runs made
                # before mixing existed still reproduce exactly.
                self._orders[pool_index] = self._permutation(
                    pool_size,
                    self.seed * 100003 + pool_index * 1000003 + self._passes[pool_index],
                )
                self._cursors[pool_index] = 0
                self._passes[pool_index] += 1
            cursor = self._cursors[pool_index]
            take = min(count - len(indices), len(self._orders[pool_index]) - cursor)
            indices.extend(self._orders[pool_index][cursor : cursor + take])
            self._cursors[pool_index] = cursor + take
        self._draws[pool_index] += len(indices)
        return indices

    def _rows(self, dataset, indices: list[int]) -> list[dict]:
        columns = dataset[indices]
        return [
            {name: columns[name][position] for name in TOKEN_COLUMNS}
            for position in range(len(indices))
        ]

    def draw_counts(self) -> dict[str, int]:
        """Replay rows served per pool so far, for the end-of-run log."""

        return {
            pool.label or "replay": count for pool, count in zip(self.pools, self._draws)
        }

    def __iter__(self):
        # Advance before yielding so an early break (max_steps) still moves the
        # epoch on, matching DataLoader(shuffle=True) reshuffling per epoch.
        epoch_seed = self.seed * 7919 + self.epoch
        self.epoch += 1
        new_order = self._permutation(len(self.new_dataset), epoch_seed)
        shuffle_generator = torch.Generator().manual_seed(epoch_seed + 1)

        for step in range(self.steps_per_epoch):
            start = step * self.new_per_batch
            new_indices = new_order[start : start + self.new_per_batch]
            rows = self._rows(self.new_dataset, new_indices)
            for pool_index, count in enumerate(self._next_counts()):
                if count == 0:
                    continue
                rows.extend(
                    self._rows(
                        self.pools[pool_index].dataset,
                        self._next_replay_indices(pool_index, count),
                    )
                )
            order = torch.randperm(len(rows), generator=shuffle_generator).tolist()
            yield self.collate_fn([rows[position] for position in order])


def build_batch_mixed_loader(
    new_dataset,
    replay_dataset,
    tokenizer,
    new_per_batch: int,
    replay_per_batch: int,
    seed: int = 1,
    replay_pools: Sequence[ReplayPool | tuple] | None = None,
) -> BatchMixedReplayLoader:
    """Build a strict-composition replay loader matching build_loader's collation.

    Pass replay_dataset for one corpus, or replay_pools (and replay_dataset=None)
    for a weighted mix of several.
    """

    tokenizer.padding_side = "left"
    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return BatchMixedReplayLoader(
        new_dataset=_keep_token_columns(new_dataset),
        replay_dataset=None if replay_dataset is None else _keep_token_columns(replay_dataset),
        replay_pools=(
            None
            if replay_pools is None
            else [
                ReplayPool(pool.label, _keep_token_columns(pool.dataset), pool.weight)
                for pool in (
                    pool if isinstance(pool, ReplayPool) else ReplayPool(*pool)
                    for pool in replay_pools
                )
            ]
        ),
        collate_fn=collate_fn,
        new_per_batch=new_per_batch,
        replay_per_batch=replay_per_batch,
        seed=seed,
    )


__all__ = ["BatchMixedReplayLoader", "ReplayPool", "build_batch_mixed_loader"]
