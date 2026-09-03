"""Deterministic per-epoch sample ordering and audit hashes."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class DeterministicEpochSampler(Sampler[int]):
    """Generate the same sample-id order for baseline and every candidate."""

    def __init__(self, sample_ids: Sequence[str], seed: int) -> None:
        self.sample_ids = [str(value) for value in sample_ids]
        if not self.sample_ids:
            raise ValueError("sample_ids must not be empty")
        self.seed = int(seed)
        self.epoch = -1
        self._indices: list[int] = []
        self._hash = ""
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self._indices = torch.randperm(len(self.sample_ids), generator=generator).tolist()
        ordered_ids = [self.sample_ids[index] for index in self._indices]
        encoded = json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._hash = hashlib.sha256(encoded).hexdigest()

    def order_hash(self) -> str:
        return self._hash

    def ordered_sample_ids(self) -> list[str]:
        return [self.sample_ids[index] for index in self._indices]

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self.sample_ids)


def seed_worker(worker_id: int) -> None:
    """Split worker-side numpy/random streams from model initialization."""

    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
