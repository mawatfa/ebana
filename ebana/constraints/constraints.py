"""
Constraints are applied after every update. They get the new and the old
weights, so they can limit the values or the size of the step. Layers take
one constraint, a list of them, or None.
"""

from abc import ABC, abstractmethod

import numpy as np


class Constraint(ABC):
    @abstractmethod
    def __call__(self, w_new, w_old):
        pass


def as_constraint(constraint):
    if constraint is None:
        return Chain([])
    if isinstance(constraint, (list, tuple)):
        return Chain(constraint)
    return constraint


class Chain(Constraint):
    def __init__(self, constraints):
        self.constraints = list(constraints)

    def __call__(self, w_new, w_old):
        for constraint in self.constraints:
            w_new = constraint(w_new, w_old)
        return w_new


class Clip(Constraint):
    def __init__(self, low=None, high=None):
        if low is not None and high is not None and low >= high:
            raise ValueError(f"need low < high, got {low} and {high}")
        self.low = low
        self.high = high

    def __call__(self, w_new, w_old):
        return np.clip(w_new, self.low, self.high)


class MaxRelativeChange(Constraint):
    """Limits every step to a fraction of the weight, so a positive weight stays positive."""

    def __init__(self, fraction):
        if fraction <= 0:
            raise ValueError(f"fraction must be positive, got {fraction}")
        self.fraction = fraction

    def __call__(self, w_new, w_old):
        max_change = self.fraction * np.abs(w_old)
        return w_old + np.clip(w_new - w_old, -max_change, max_change)
