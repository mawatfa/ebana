import numpy as np
import pickle
from .base import BaseOptimizer

class Adam(BaseOptimizer):
    def __init__(self, model, beta, gamma_v=0.5, gamma_s=0.5, eps=1e-8):
        self.gamma_v = gamma_v
        self.gamma_s = gamma_s
        self.eps = eps
        self.k = 0
        super().__init__(model, beta)

    def make_optimizer_variables(self):
        self.v, self.s = {}, {}
        for layer in self.model.computation_graph:
            if layer.trainable:
                self.v[layer.name] = np.zeros(layer.shape)
                self.s[layer.name] = np.zeros(layer.shape)

    def step(self):
        self.k += 1
        for layer in self.model.computation_graph:
            if layer.trainable:

                update = layer.accumulated_gradient / (self.model.batch_size * self.beta)

                self.v[layer.name] = self.gamma_v * self.v[layer.name] + (1 - self.gamma_v) * update
                self.v[layer.name] /= (1 - self.gamma_v ** self.k)

                self.s[layer.name] = self.gamma_s * self.s[layer.name] + (1 - self.gamma_s) * (update ** 2)
                self.s[layer.name] /= (1 - self.gamma_s ** self.k)

                gradient = self.v[layer.name] / (np.sqrt(self.s[layer.name]) + self.eps)
                layer.weight_update_func(gradient, self.model.epoch_num, self.model.batch_num, self.model.num_batches)

    def save_state(self, filepath):
        optimizer_state = {}
        for layer in self.model.computation_graph:
            layer_variables = layer.get_variables()
            if layer_variables != {}:
                optimizer_state[layer.name] = layer_variables
                if layer.name in self.v:
                    optimizer_state[layer.name]["velocity"] = self.v[layer.name]
                    optimizer_state[layer.name]["acceleration"] = self.s[layer.name]
        optimizer_state["k"] = self.k
        with open(filepath, "wb") as handle:
            pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _set_optimizer_state(self, optimizer_state):
        super()._set_optimizer_state(optimizer_state)
        self.k = optimizer_state["k"]
        for layer in self.model.computation_graph:
            if layer.name in optimizer_state and "velocity" in optimizer_state[layer.name]:
                self.v[layer.name] = optimizer_state[layer.name]["velocity"]
                self.s[layer.name] = optimizer_state[layer.name]["acceleration"]
