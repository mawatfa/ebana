import numpy as np

from .dataset import Dataset


class DataLoader:
    """
    Yields the full batches of an epoch. When shuffling, the order depends
    only on (seed, epoch), so a resumed run sees the same batches.
    """

    def __init__(self, dataset, batch_size, shuffle=False, seed=None):
        if not 1 <= batch_size <= len(dataset):
            raise ValueError(f"batch_size must be between 1 and {len(dataset)}, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def batches(self, epoch=0):
        n = len(self.dataset)
        if self.shuffle:
            seed = None if self.seed is None else [self.seed, epoch]
            order = np.random.default_rng(seed).permutation(n)
        else:
            order = np.arange(n)
        for b in range(len(self)):
            yield self.dataset[order[b * self.batch_size : (b + 1) * self.batch_size]]
