from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    name: str

    @abstractmethod
    def __call__(self, y, target):
        """The score of each sample. y and target are (samples, units)."""


class BinaryAccuracy(Metric):
    """1 when every output is on the same side of the threshold as its target."""

    name = "binary_accuracy"

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, y, target):
        return np.all((y >= self.threshold) == (target >= self.threshold), axis=-1).astype(float)


class CategoricalAccuracy(Metric):
    """1 when the largest output is the target class."""

    name = "categorical_accuracy"

    def __call__(self, y, target):
        return (np.argmax(y, axis=-1) == np.argmax(target, axis=-1)).astype(float)
