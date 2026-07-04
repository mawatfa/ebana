import numpy as np
from .base import BaseOptimizer


class Adadelta(BaseOptimizer):
    """The learning rate still scales the update; lr=1 gives the original method."""

    slots = ("grad_sq", "update_sq")

    def __init__(self, model, rho=0.95, eps=1e-6, transforms=()):
        self.rho = rho
        self.eps = eps
        super().__init__(model, transforms)

    def compute_update(self, grad, state):
        state["grad_sq"] = self.rho * state["grad_sq"] + (1 - self.rho) * grad**2
        update = np.sqrt(state["update_sq"] + self.eps) / np.sqrt(state["grad_sq"] + self.eps) * grad
        state["update_sq"] = self.rho * state["update_sq"] + (1 - self.rho) * update**2
        return update
