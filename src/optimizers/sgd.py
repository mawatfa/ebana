from .base import BaseOptimizer


class SGD(BaseOptimizer):
    def step(self):
        for layer in self.model.computation_graph:
            if layer.trainable:
                gradient = layer.accumulated_gradient / (self.model.batch_size * self.beta)
                layer.weight_update_func(gradient, self.model.epoch_num, self.model.batch_num)

