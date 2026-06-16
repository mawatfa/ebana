import pickle
from abc import ABC, abstractmethod

import numpy as np

CHECKPOINT_VERSION = 1


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
            layer.name: {slot: np.zeros(layer.shape) for slot in self.slots}
            for layer in self._trainable_layers()
        }

    def _trainable_layers(self):
        return [layer for layer in self.model.computation_graph if layer.trainable]

    @abstractmethod
    def compute_update(self, grad, state) -> np.ndarray:
        """The update of one layer. state holds its slots and can be changed in place."""

    def step(self):
        for layer in self._trainable_layers():
            grad = layer.accumulated_gradient / (self.model.batch_size * self.beta)
            update = self.compute_update(grad, self.state[layer.name])
            layer.weight_update_func(update, self.iterations)
        self.iterations += 1

    # checkpointing <<<
    def save_state(self, filepath):
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "optimizer": type(self).__name__,
            "iterations": self.iterations,
            "state": self.state,
            "variables": {
                layer.name: layer.get_variables() for layer in self.model.computation_graph
            },
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

        names = {layer.name for layer in self.model.computation_graph}
        if set(checkpoint["variables"]) != names:
            raise ValueError(
                f"{filepath}: layers {sorted(checkpoint['variables'])} do not match the model's {sorted(names)}"
            )
        for layer_name, slots in self.state.items():
            saved = checkpoint["state"].get(layer_name, {})
            for slot, array in slots.items():
                if slot not in saved or saved[slot].shape != array.shape:
                    raise ValueError(
                        f"{filepath}: missing or mis-shaped optimizer slot '{slot}' for layer '{layer_name}'"
                    )

        for layer in self.model.computation_graph:
            layer.set_variables(checkpoint["variables"][layer.name])
        self.state = checkpoint["state"]
        self.iterations = checkpoint["iterations"]

    # >>>
