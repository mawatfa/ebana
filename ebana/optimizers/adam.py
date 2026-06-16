import numpy as np
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    # the decay rates are called gamma so they aren't confused with the beta of EP

    slots = ("m", "v")

    def __init__(self, model, beta, gamma_m=0.9, gamma_v=0.999, eps=1e-8):
        self.gamma_m = gamma_m
        self.gamma_v = gamma_v
        self.eps = eps
        super().__init__(model, beta)

    def compute_update(self, grad, state):
        state["m"] = self.gamma_m * state["m"] + (1 - self.gamma_m) * grad
        state["v"] = self.gamma_v * state["v"] + (1 - self.gamma_v) * grad**2
        t = self.iterations + 1
        m_hat = state["m"] / (1 - self.gamma_m**t)
        v_hat = state["v"] / (1 - self.gamma_v**t)
        return m_hat / (np.sqrt(v_hat) + self.eps)
