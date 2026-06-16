from .base import BaseOptimizer


class SGD(BaseOptimizer):
    def compute_update(self, grad, state):
        return grad
