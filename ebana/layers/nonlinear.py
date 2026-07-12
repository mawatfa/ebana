import numpy as np

from ..devices import BidirectionalAmplifier, IdealDiode, VoltageSource
from ..simulation import GROUND
from .base import Layer, net_names


class Diode(Layer):
    """
    One diode per net, to its own bias source. "down" conducts from the net
    to the bias when the net is above it, "up" from the bias to the net when
    the net is below it. The nets pass through unchanged.
    """

    def __init__(self, name, direction, model=None, bias_voltage=0.0, trainable=False, lr=0.0,
                 constraint=None, grad_func=None, update_func=None):
        super().__init__(name)
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
        self.direction = direction
        self.model = model or IdealDiode()
        self.bias_voltage = bias_voltage
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable,
                            grad_func=grad_func, update_func=update_func)

    def build(self, inputs):
        (x,) = inputs
        self.bias = self.add_parameter("bias", np.full(x.shape, self.bias_voltage, dtype=float),
                                       VoltageSource(), **self.options)
        return x

    def stamp(self, block, data, sample_index):
        sources = []
        for i, (net, bias) in enumerate(zip(np.ravel(self.output), self.bias.value.ravel()), start=1):
            node = block.node(f"n{i}")
            anode, cathode = (net, node) if self.direction == "down" else (node, net)
            self.model.stamp(block, f"d{i}", anode, cathode)
            sources.append(self.bias.device.stamp(block, i, node, GROUND, bias))
        self.bias.elements = block.array(sources, self.bias.shape)


def shared_gain_update(parameter, update, step):
    """Keeps a single gain for all the stages of an Amplifier."""
    return np.full(parameter.shape, np.mean(parameter.value - parameter.lr(step) * update))


class Amplifier(Layer):
    """One amplifier stage per net, each driving a new net."""

    def __init__(self, name, gain=1.0, device=None, trainable=False, lr=0.0, constraint=None,
                 grad_func=None, update_func=shared_gain_update):
        super().__init__(name)
        self.initial_gain = gain
        self.device = device or BidirectionalAmplifier()
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable,
                            grad_func=grad_func, update_func=update_func)

    def build(self, inputs):
        (x,) = inputs
        self.inputs = x
        self.gain = self.add_parameter("gain", np.full(x.shape, self.initial_gain, dtype=float),
                                       self.device, **self.options)
        return net_names(x.shape, f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        p = self.gain
        stages = [p.device.stamp(block, i, a, b, gain)
                  for i, (a, b, gain) in enumerate(zip(np.ravel(self.inputs), np.ravel(self.output), p.value.ravel()), start=1)]
        p.elements = block.array(stages, p.shape)
