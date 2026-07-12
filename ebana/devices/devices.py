"""
Devices are elements with a trainable value (conductance, source value, gain).
A device knows how to stamp itself, what to measure after a solve, and how two
measurements give the gradient: dE/dp in state b minus dE/dp in state a, where
E is the circuit's energy. EP passes (free, nudged) states and symmetric EP
(-beta, +beta) states.

Element models are elements without a trainable value, like a diode.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..simulation import GROUND, Element


# measurements <<<
def _node_voltages(sim, elements, terminal):
    # read each net once
    nets = np.vectorize(lambda e: e.nodes[terminal], otypes=[object])(elements)
    unique, inverse = np.unique(nets.astype(str), return_inverse=True)
    values = np.zeros(len(unique))
    grounded = unique == GROUND
    values[~grounded] = sim.voltages(unique[~grounded].astype(object))
    return values[inverse].reshape(nets.shape)


def voltage_across(sim, elements):
    elements = np.asarray(elements, dtype=object)
    return _node_voltages(sim, elements, 0) - _node_voltages(sim, elements, 1)
# >>>


# devices <<<
class Device(ABC):
    @abstractmethod
    def stamp(self, block, id, a, b, value):
        """Add one element and return its handle. For an amplifier, a is the input and b the output."""

    @abstractmethod
    def measure(self, sim, elements):
        pass

    @abstractmethod
    def gradient(self, a, b):
        pass

    def spice_value(self, value):
        return value

    def alter(self, sim, elements, value):
        sim.alter(elements, self.spice_value(np.asarray(value, dtype=float)))


class Resistor(Device):
    """The value is the conductance."""

    def __init__(self, decimals=4):
        self.decimals = decimals    # rounding of the resistance in the netlist

    def spice_value(self, value):
        return 1 / value if self.decimals is None else np.round(1 / value, self.decimals)

    def stamp(self, block, id, a, b, value):
        return block.resistor(id, a, b, self.spice_value(value))

    def measure(self, sim, elements):
        return voltage_across(sim, elements)

    def gradient(self, a, b):
        # dE/dG = V^2 / 2
        return (b**2 - a**2) / 2


class VoltageSource(Device):
    def stamp(self, block, id, a, b, value):
        return block.voltage_source(id, a, b, value)

    def measure(self, sim, elements):
        return sim.currents(elements)

    def gradient(self, a, b):
        # dE/dV = -I, the branch current flows into the + terminal
        return a - b


class CurrentSource(Device):
    def stamp(self, block, id, a, b, value):
        return block.current_source(id, a, b, value)

    def measure(self, sim, elements):
        return voltage_across(sim, elements)

    def gradient(self, a, b):
        # dE/dI = V
        return b - a


class BidirectionalAmplifier(Device):
    """
    V(out) = shift + gain * (V(in) - shift). The input draws the output
    current divided by the gain, so the nudge also travels backwards.

    Because of that division, the gradients of everything before the
    amplifier come out scaled by 1/gain^2. The gain itself has no gradient
    that can be measured locally; train it with a grad_func.
    """

    def __init__(self, shift=0.0):
        self.shift = shift

    def stamp(self, block, id, a, b, value):
        stage = block.behavioral_voltage(id, b, GROUND, f"{{{self.shift} + {value} * (V({a}) - {self.shift})}}")
        block.behavioral_current(f"i{id}", a, GROUND, f"{{{-1 / value} * I({stage.name})}}")
        return Element(name=stage.name, kind=stage.kind, nodes=(a, b), branch=True)

    def measure(self, sim, elements):
        inputs = np.vectorize(lambda e: e.nodes[0], otypes=[object])(elements)
        return -sim.currents(elements) * sim.voltages(inputs)

    def alter(self, sim, elements, value):
        raise NotImplementedError("the gain is part of the expressions; rebuild the netlist instead")

    def gradient(self, a, b):
        raise NotImplementedError("the amplifier gain needs a grad_func")
# >>>


# element models <<<
class ElementModel(ABC):
    @abstractmethod
    def stamp(self, block, id, anode, cathode):
        pass


class IdealDiode(ElementModel):
    """Piecewise linear: ron above vth, roff below."""

    def __init__(self, vth=0.3, ron=1.0, roff=1e20):
        self.vth, self.ron, self.roff = vth, ron, roff

    def stamp(self, block, id, anode, cathode):
        v = f"(V({anode}) - V({cathode}) - {self.vth})"
        return block.behavioral_current(id, anode, cathode, f"{{ max(({v}/{self.roff}), ({v}/{self.ron})) }}")


class SpiceDiode(ElementModel):
    """A SPICE diode, from model parameters or from a library at `path` that defines `model`."""

    def __init__(self, model="d", path=None, **params):
        if path is None and not params:
            raise ValueError("give the model parameters or a library path")
        self.model, self.path, self.params = model, path, params

    def stamp(self, block, id, anode, cathode):
        if self.path is not None:
            block.include(self.path)
            name = self.model
        else:
            name = block.model(self.model, "D", **self.params)
        return block.diode(id, anode, cathode, model=name)
# >>>
