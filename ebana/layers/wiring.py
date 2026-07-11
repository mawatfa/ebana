import numpy as np

from .base import Layer, net_names


class _Wiring(Layer):
    def stamp(self, block, data, sample_index):
        pass


class Reshape(_Wiring):
    def __init__(self, name, target_shape):
        super().__init__(name)
        self.target_shape = tuple(target_shape)

    def build(self, inputs):
        (x,) = inputs
        return x.reshape(self.target_shape)


class Concatenate(_Wiring):
    def __init__(self, name, axis=0):
        super().__init__(name)
        self.axis = axis

    def build(self, inputs):
        return np.concatenate(inputs, axis=self.axis)


class Stack(_Wiring):
    def __init__(self, name, axis=0):
        super().__init__(name)
        self.axis = axis

    def build(self, inputs):
        return np.stack(inputs, axis=self.axis)


class Join(Layer):
    """Ties the nets of every parent to the first parent's nets with 0 V sources."""

    def build(self, inputs):
        if len({x.shape for x in inputs}) != 1:
            raise ValueError(f"Join '{self.name}' needs inputs of the same shape, got {[x.shape for x in inputs]}")
        self.inputs = inputs
        return inputs[0]

    def stamp(self, block, data, sample_index):
        first = np.ravel(self.inputs[0])
        for p, other in enumerate(self.inputs[1:], start=1):
            for i, (a, b) in enumerate(zip(first, np.ravel(other)), start=1):
                if a != b:
                    block.voltage_source(f"{p}_{i}", b, a, 0)


class Ammeter(Layer):
    """A 0 V source in series with every net. currents(sim) gives the current from input to output."""

    def build(self, inputs):
        (x,) = inputs
        self.inputs = x
        return net_names(x.shape, f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        meters = [block.voltage_source(i, a, b, 0)
                  for i, (a, b) in enumerate(zip(np.ravel(self.inputs), np.ravel(self.output)), start=1)]
        self.meters = block.array(meters, self.output_shape)

    def currents(self, sim):
        return sim.currents(self.meters)
