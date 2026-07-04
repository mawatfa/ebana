from .base import BaseOptimizer


class SGDMomentum(BaseOptimizer):
    """Heavy ball or Nesterov momentum."""

    slots = ("velocity",)

    def __init__(self, model, momentum=0.9, nesterov=False, transforms=()):
        self.momentum = momentum
        self.nesterov = nesterov
        super().__init__(model, transforms)

    def compute_update(self, grad, state):
        state["velocity"] = self.momentum * state["velocity"] + grad
        if self.nesterov:
            return grad + self.momentum * state["velocity"]
        return state["velocity"]
