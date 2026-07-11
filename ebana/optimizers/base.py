from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
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
