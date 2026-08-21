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
"""

from __future__ import annotations

import torch
from transformers import DataCollatorForTokenClassification

TOKEN_COLUMNS = ["input_ids", "labels", "attention_mask"]


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


class BatchMixedReplayLoader:
    """Iterable of micro-batches with a fixed new/replay composition."""

    def __init__(
        self,
        new_dataset,
        replay_dataset,
        collate_fn,
        new_per_batch: int,
        replay_per_batch: int,
        seed: int = 1,
    ) -> None:
        if new_per_batch <= 0:
            raise ValueError("new_per_batch must be >= 1")
        if replay_per_batch <= 0:
            raise ValueError("replay_per_batch must be >= 1; use build_loader for no replay")
        if len(replay_dataset) == 0:
            raise ValueError("replay pool is empty")

        self.new_dataset = new_dataset
        self.replay_dataset = replay_dataset
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
        self._replay_order: list[int] = []
        self._replay_cursor = 0
        self._replay_passes = 0

    @property
    def new_samples_per_epoch(self) -> int:
        return self.steps_per_epoch * self.new_per_batch

    @property
    def replay_samples_per_epoch(self) -> int:
        return self.steps_per_epoch * self.replay_per_batch

    def describe(self) -> str:
        dropped = len(self.new_dataset) - self.new_samples_per_epoch
        replay_epochs = self.replay_samples_per_epoch / len(self.replay_dataset)
        new_tokens = _mean_supervised_tokens(self.new_dataset)
        replay_tokens = _mean_supervised_tokens(self.replay_dataset)
        row_share = self.replay_per_batch / (self.new_per_batch + self.replay_per_batch)
        token_total = self.new_per_batch * new_tokens + self.replay_per_batch * replay_tokens
        token_share = self.replay_per_batch * replay_tokens / token_total if token_total else 0.0
        return (
            f"batch-level replay: {self.new_per_batch} new + {self.replay_per_batch} replay "
            f"per micro-batch (micro-batch={self.new_per_batch + self.replay_per_batch}), "
            f"{self.steps_per_epoch} steps/epoch, "
            f"{self.new_samples_per_epoch} new-task rows/epoch (dropped {dropped} tail rows), "
            f"{self.replay_samples_per_epoch} replay rows/epoch from a pool of "
            f"{len(self.replay_dataset)} (cycled {replay_epochs:.2f}x per epoch)\n"
            f"  replay share of rows={row_share:.4f}, of supervised tokens={token_share:.4f} "
            f"(mean supervised tokens: new={new_tokens:.1f}, replay={replay_tokens:.1f})"
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _permutation(self, size: int, seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        return torch.randperm(size, generator=generator).tolist()

    def _next_replay_indices(self) -> list[int]:
        """Take the next replay_per_batch indices, reshuffling at pool wrap."""

        indices: list[int] = []
        while len(indices) < self.replay_per_batch:
            if self._replay_cursor >= len(self._replay_order):
                # A fresh permutation per pass keeps the pool from pairing the
                # same replay rows with the same new-task rows every cycle.
                self._replay_order = self._permutation(
                    len(self.replay_dataset), self.seed * 100003 + self._replay_passes
                )
                self._replay_cursor = 0
                self._replay_passes += 1
            take = min(self.replay_per_batch - len(indices), len(self._replay_order) - self._replay_cursor)
            indices.extend(self._replay_order[self._replay_cursor : self._replay_cursor + take])
            self._replay_cursor += take
        return indices

    def _rows(self, dataset, indices: list[int]) -> list[dict]:
        columns = dataset[indices]
        return [
            {name: columns[name][position] for name in TOKEN_COLUMNS}
            for position in range(len(indices))
        ]

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
            replay_indices = self._next_replay_indices()
            rows = self._rows(self.new_dataset, new_indices) + self._rows(
                self.replay_dataset, replay_indices
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
) -> BatchMixedReplayLoader:
    """Build a strict-composition replay loader matching build_loader's collation."""

    tokenizer.padding_side = "left"
    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return BatchMixedReplayLoader(
        new_dataset=_keep_token_columns(new_dataset),
        replay_dataset=_keep_token_columns(replay_dataset),
        collate_fn=collate_fn,
        new_per_batch=new_per_batch,
        replay_per_batch=replay_per_batch,
        seed=seed,
    )


__all__ = ["BatchMixedReplayLoader", "build_batch_mixed_loader"]
