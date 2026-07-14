"""
Weight initializers. The weights are conductances, so the random ones draw
from a positive range [low, high]. One seeded initializer shared by several
layers gives each layer different weights and the same model every run.
"""

from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape):
        pass


class Constant(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return np.full(shape, self.value, dtype=float)


class _RandomInitializer(Initializer):
    def __init__(self, low, high, seed=None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)


class RandomUniform(_RandomInitializer):
    def __call__(self, shape):
        return self.rng.uniform(self.low, self.high, size=shape)


class LogUniform(_RandomInitializer):
    """Uniform in log space."""

    def __call__(self, shape):
        return np.exp(self.rng.uniform(np.log(self.low), np.log(self.high), size=shape))


class GlorotUniform(_RandomInitializer):
    """Uniform in [low, high / sqrt(fan_in + fan_out + 1)], so larger layers get smaller conductances."""

    def _limit(self, shape):
        fan_in, fan_out = shape
        return self.high / np.sqrt(fan_in + fan_out + 1)

    def __call__(self, shape):
        return self.rng.uniform(self.low, self._limit(shape), size=shape)


class GlorotLogUniform(GlorotUniform):
    def __call__(self, shape):
        return np.exp(self.rng.uniform(np.log(self.low), np.log(self._limit(shape)), size=shape))


class HeUniform(_RandomInitializer):
    """Uniform in [low, high / sqrt(fan_in + 1)]."""

    def __call__(self, shape):
        return self.rng.uniform(self.low, self.high / np.sqrt(shape[0] + 1), size=shape)
