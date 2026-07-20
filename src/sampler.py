from __future__ import annotations

import math
import random

from torch.utils.data import Sampler


class WeightedSourceBatchSampler(Sampler[list[int]]):
    def __init__(self, labels: list[str], batch_size: int, pano_ratio: float = 0.75,
                 num_batches: int | None = None, seed: int = 42):
        if batch_size < 2:
            raise ValueError("Kaynak dengesi için batch_size en az 2 olmalıdır.")
        if not 0 < pano_ratio < 1:
            raise ValueError("pano_ratio 0 ile 1 arasında olmalıdır.")
        self.pano = [i for i, label in enumerate(labels) if label == "panoramic"]
        self.bitewing = [i for i, label in enumerate(labels) if label == "bitewing"]
        if not self.pano or not self.bitewing:
            raise ValueError("Dengeli sampler için iki kaynakta da örnek bulunmalıdır.")
        self.batch_size = batch_size
        self.n_pano = max(1, min(batch_size - 1, round(batch_size * pano_ratio)))
        self.num_batches = num_batches or math.ceil(len(labels) / batch_size)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch = rng.choices(self.pano, k=self.n_pano)
            batch += rng.choices(self.bitewing, k=self.batch_size - self.n_pano)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

