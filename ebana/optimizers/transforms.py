"""Gradient transforms, applied before the optimizer's update."""

import numpy as np


class ClipByValue:
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, parameter, grad):
        return np.clip(grad, -self.limit, self.limit)


class ClipByNorm:
    """Limits the L2 norm of each parameter's gradient."""

    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, parameter, grad):
        norm = np.linalg.norm(grad)
        return grad * (self.max_norm / norm) if norm > self.max_norm else grad
