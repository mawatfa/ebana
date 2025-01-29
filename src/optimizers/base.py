import pickle
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self, model, beta):
        self.model = model
        self.beta = beta
        self.make_optimizer_variables()

    def make_optimizer_variables(self):
        """Create optimizer-specific variables. Override in subclasses if needed."""
        pass

    @abstractmethod
    def step(self):
        """Apply the update rule. Must be implemented in subclasses."""
        pass

    def save_state(self, filepath):
        optimizer_state = {}
        for layer in self.model.computation_graph:
            layer_variables = layer.get_variables()
            if layer_variables != {}:
                optimizer_state[layer.name] = layer_variables
        with open(filepath, "wb") as handle:
            pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath):
        with open(filepath, "rb") as handle:
            optimizer_state = pickle.load(handle)
        self._set_optimizer_state(optimizer_state)

    def _set_optimizer_state(self, optimizer_state):
        """Restore optimizer state from the saved state. Override in subclasses if needed."""
        for layer in self.model.computation_graph:
            if layer.name in optimizer_state:
                layer.set_variables(optimizer_state[layer.name])
