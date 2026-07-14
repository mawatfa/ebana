from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """
    Subclasses list the arrays they keep for each parameter in `slots` and
    implement compute_update. The transforms are applied to the gradients
    first.
    """

    slots = ()

    def __init__(self, model, transforms=()):
        self.model = model
        self.transforms = list(transforms)
        self.iterations = 0
        self.state = {
            p.full_name: {slot: np.zeros(p.shape) for slot in self.slots}
            for p in self.model.trainable_parameters()
        }

    @abstractmethod
    def compute_update(self, grad, state):
        """The update of one parameter. state holds its slots and can be changed in place."""

    def step(self):
        for p in self.model.trainable_parameters():
            grad = p.grad
            for transform in self.transforms:
                grad = transform(p, grad)
            update = self.compute_update(grad, self.state[p.full_name])
            p.apply_update(update, self.iterations)
        self.iterations += 1

    # state <<<
    def state_dict(self):
        return {
            "optimizer": type(self).__name__,
            "iterations": self.iterations,
            "slots": {name: {k: v.copy() for k, v in slots.items()} for name, slots in self.state.items()},
        }

    def load_state_dict(self, state):
        if state["optimizer"] != type(self).__name__:
            raise ValueError(f"the state was saved by {state['optimizer']}, not {type(self).__name__}")
        if set(state["slots"]) != set(self.state):
            raise ValueError(f"the state is for the parameters {sorted(state['slots'])}, not {sorted(self.state)}")
        self.state = {name: {k: np.array(v, dtype=float) for k, v in slots.items()}
                      for name, slots in state["slots"].items()}
        self.iterations = state["iterations"]
    # >>>


class SGD(Optimizer):
    def compute_update(self, grad, state):
        return grad


class SGDMomentum(Optimizer):
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


class Adam(Optimizer):
    # the decay rates are called gamma so they aren't confused with the beta of EP

    slots = ("m", "v")

    def __init__(self, model, gamma_m=0.9, gamma_v=0.999, eps=1e-8, transforms=()):
        self.gamma_m = gamma_m
        self.gamma_v = gamma_v
        self.eps = eps
        super().__init__(model, transforms)

    def compute_update(self, grad, state):
        state["m"] = self.gamma_m * state["m"] + (1 - self.gamma_m) * grad
        state["v"] = self.gamma_v * state["v"] + (1 - self.gamma_v) * grad**2
        t = self.iterations + 1
        m_hat = state["m"] / (1 - self.gamma_m**t)
        v_hat = state["v"] / (1 - self.gamma_v**t)
        return m_hat / (np.sqrt(v_hat) + self.eps)


class Adadelta(Optimizer):
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


# gradient transforms <<<
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
# >>>
