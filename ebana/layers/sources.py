import numpy as np

from ..devices import VoltageSource
from ..simulation import GROUND
from .base import SourceLayer, net_names


class Input(SourceLayer):
    """Sources set by the dataset input with the same name. device can also be CurrentSource()."""

    def __init__(self, name, units, device=None, save_sim_data=False):
        self.shape = (units,) if np.isscalar(units) else tuple(units)
        self.device = device or VoltageSource()
        super().__init__(name, save_sim_data=save_sim_data)

    def build(self, inputs):
        return net_names(self.shape, f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        values = np.ravel(data[self.name][sample_index])
        nets = np.ravel(self.output)
        if len(values) != len(nets):
            raise ValueError(f"input '{self.name}' expects {len(nets)} values per sample, got {len(values)}")
        self.sources = block.array(
            [self.device.stamp(block, i + 1, net, GROUND, v) for i, (net, v) in enumerate(zip(nets, values))],
            self.shape,
        )

    def record(self, sim):
        return {**super().record(sim), "sources": {"measurement": self.device.measure(sim, self.sources)}}

    def record_spec(self):
        return {**super().record_spec(), "sources": {"measurement": self.shape}}


class Bias(SourceLayer):
    """Fixed sources. Their values are the parameter `value`, which can be trained."""

    def __init__(self, name, values, device=None, trainable=False, lr=0.0,
                 constraint=None, grad_func=None, update_func=None, save_sim_data=False):
        self.initial = np.atleast_1d(np.asarray(values, dtype=float))
        self.device = device or VoltageSource()
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable,
                            grad_func=grad_func, update_func=update_func)
        super().__init__(name, save_sim_data=save_sim_data)

    def build(self, inputs):
        self.value = self.add_parameter("value", self.initial, self.device, **self.options)
        return net_names(self.initial.shape, f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        p = self.value
        nets, values = np.ravel(self.output), p.value.ravel()
        p.elements = block.array(
            [p.device.stamp(block, i + 1, net, GROUND, v) for i, (net, v) in enumerate(zip(nets, values))],
            p.shape,
        )
