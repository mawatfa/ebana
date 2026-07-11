from abc import ABC, abstractmethod

import numpy as np

from ..simulation import GROUND
from .base import Layer


class Readout(ABC):
    """How outputs are read from the output nets, and which currents push them."""

    @abstractmethod
    def units(self, nets):
        pass

    @abstractmethod
    def value(self, voltages):
        pass

    @abstractmethod
    def adjoint(self, force):
        """Current to inject into each net for a force on the outputs."""


class Direct(Readout):
    def units(self, nets):
        return nets

    def value(self, voltages):
        return voltages

    def adjoint(self, force):
        return force


class Differential(Readout):
    """Output i is V(2i) - V(2i + 1)."""

    def units(self, nets):
        if nets % 2:
            raise ValueError(f"a differential readout needs an even number of nets, got {nets}")
        return nets // 2

    def value(self, voltages):
        pairs = voltages.reshape(-1, 2)
        return pairs[:, 0] - pairs[:, 1]

    def adjoint(self, force):
        return np.stack([force, -force], axis=1).ravel()


class CurrentOutput(Layer):
    """Reads the input nets, and nudges them with current sources (zero in the free phase)."""

    def __init__(self, name, readout=None):
        super().__init__(name)
        self.readout = readout or Direct()

    def build(self, inputs):
        (x,) = inputs
        if x.ndim != 1:
            raise ValueError(f"output '{self.name}' needs a 1-D input, got shape {x.shape}")
        self.units = self.readout.units(len(x))
        return x

    def stamp(self, block, data, sample_index):
        self.sources = block.array(
            [block.current_source(i + 1, net, GROUND, -0.0) for i, net in enumerate(self.output)],
            self.output_shape,
        )

    def read(self, sim):
        return self.readout.value(sim.voltages(self.output))

    def nudge(self, force):
        """The sources and the values that push the outputs with `force` (A)."""
        # the source current flows out of the net
        return self.sources, -self.readout.adjoint(np.asarray(force, dtype=float))
