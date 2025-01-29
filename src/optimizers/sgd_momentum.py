import numpy as np
import pickle
from .base import BaseOptimizer

class SGDMomentum(BaseOptimizer):
    def __init__(self, model, beta, momentum=0.9):
        self.momentum = momentum
        super().__init__(model, beta)

    def make_optimizer_variables(self):
        self.v = {}
        for layer in self.model.computation_graph:
            if layer.trainable:
                self.v[layer.name] = np.zeros(layer.shape)

    def step(self):
        for layer in self.model.computation_graph:
            if layer.trainable:
                update = layer.accumulated_gradient / (self.model.batch_size * self.beta)
                self.v[layer.name] = self.momentum * self.v[layer.name] + (1 - self.momentum) * update
                layer.weight_update_func(self.v[layer.name], self.model.epoch_num, self.model.batch_num)

    # def step(self):
    #     for layer in self.model.computation_graph:
    #         if layer.trainable:
    #             update = layer.accumulated_gradient / (self.model.batch_size * self.beta)
    #             prev_vel = self.v[layer.name].copy()
    #             self.v[layer.name] = self.momentum * self.v[layer.name] + (1 - self.momentum) * update
    #             nesterov_update = self.momentum * prev_vel + self.v[layer.name]
    #             layer.weight_update_func(nesterov_update, self.model.epoch_num, self.model.batch_num)

    def save_state(self, filepath):
        optimizer_state = {}
        for layer in self.model.computation_graph:
            layer_variables = layer.get_variables()
            if layer_variables != {}:
                optimizer_state[layer.name] = layer_variables
                if layer.name in self.v:
                    optimizer_state[layer.name]["velocity"] = self.v[layer.name]
        with open(filepath, "wb") as handle:
            pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _set_optimizer_state(self, optimizer_state):
        super()._set_optimizer_state(optimizer_state)
        for layer in self.model.computation_graph:
            if layer.name in optimizer_state and "velocity" in optimizer_state[layer.name]:
                self.v[layer.name] = optimizer_state[layer.name]["velocity"]
