import pickle
from abc import ABC, abstractmethod

import numpy as np

CHECKPOINT_VERSION = 2


class BaseOptimizer(ABC):
    """
    Subclasses list the arrays they keep for each parameter in `slots` and
    implement compute_update.
    """

    slots = ()

    def __init__(self, model, beta):
        self.model = model
        self.beta = beta
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
            grad = p.grad / (self.model.batch_size * self.beta)
            update = self.compute_update(grad, self.state[p.full_name])
            p.apply_update(update, self.iterations)
        self.iterations += 1

    # checkpointing <<<
    def save_state(self, filepath):
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "optimizer": type(self).__name__,
            "iterations": self.iterations,
            "state": self.state,
            "parameters": {p.full_name: p.value for p in self.model.parameters()},
        }
        with open(filepath, "wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath):
        with open(filepath, "rb") as handle:
            checkpoint = pickle.load(handle)

        if not isinstance(checkpoint, dict) or checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"{filepath} is not an optimizer checkpoint")
        if checkpoint["optimizer"] != type(self).__name__:
            raise ValueError(f"{filepath} was saved by {checkpoint['optimizer']}, not {type(self).__name__}")

        params = {p.full_name: p for p in self.model.parameters()}
        saved_values = checkpoint["parameters"]
        if set(saved_values) != set(params):
            raise ValueError(f"{filepath} has the parameters {sorted(saved_values)}, not {sorted(params)}")

        for name, p in params.items():
            p.value = np.array(saved_values[name], dtype=float)
        self.state = checkpoint["state"]
        self.iterations = checkpoint["iterations"]

    # >>>
