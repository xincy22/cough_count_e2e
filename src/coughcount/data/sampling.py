from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class BalancedSampler(Sampler[list[int]]):
    def __init__(
        self,
        pos_idx: Sequence[int],
        neg_idx: Sequence[int],
        *,
        batch_size: int,
        pos_fraction: float = 0.5,
        drop_last: bool = False,
        seed: int = 0,
        epoch_size: int | None = None,
    ) -> None:
        if not (0.0 < pos_fraction < 1.0):
            raise ValueError("pos_fraction must be in (0, 1).")
        if batch_size <= 1:
            raise ValueError("batch_size must be > 1.")
        if len(pos_idx) == 0:
            raise ValueError("pos_idx is empty.")
        if len(neg_idx) == 0:
            raise ValueError("neg_idx is empty.")

        self.pos_idx = np.asarray(list(pos_idx), dtype=np.int64)
        self.neg_idx = np.asarray(list(neg_idx), dtype=np.int64)

        self.batch_size = int(batch_size)
        self.pos_per_batch = max(1, int(round(self.batch_size * float(pos_fraction))))
        self.neg_per_batch = self.batch_size - self.pos_per_batch

        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(int(seed))

        if epoch_size is None:
            epoch_size = len(self.neg_idx)
        self.epoch_size = int(epoch_size)

        self.num_batches = self.epoch_size // self.batch_size
        if not self.drop_last and (self.epoch_size % self.batch_size) != 0:
            self.num_batches += 1

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        pos = self.pos_idx.copy()
        neg = self.neg_idx.copy()
        self.rng.shuffle(pos)
        self.rng.shuffle(neg)

        pos_ptr = 0
        neg_ptr = 0

        for _ in range(self.num_batches):
            if self.drop_last and (pos_ptr == -1):
                break

            if pos_ptr + self.pos_per_batch > len(pos):
                self.rng.shuffle(pos)
                pos_ptr = 0
            if neg_ptr + self.neg_per_batch > len(neg):
                self.rng.shuffle(neg)
                neg_ptr = 0

            batch = np.concatenate(
                [
                    pos[pos_ptr : pos_ptr + self.pos_per_batch],
                    neg[neg_ptr : neg_ptr + self.neg_per_batch],
                ]
            )
            pos_ptr += self.pos_per_batch
            neg_ptr += self.neg_per_batch

            self.rng.shuffle(batch)
            yield batch.tolist()
