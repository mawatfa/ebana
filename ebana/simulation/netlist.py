"""
Netlist construction. Each layer adds its elements through its own Block and
keeps the returned Element handles, which are used later to alter elements and
read their currents. Names are prefixed with the block name, so the netlist
stays flat.
"""

from dataclasses import dataclass

import numpy as np
from pycircuitsim.netlisting import Circuit

GROUND = "0"
TEMPERATURE = 27


@dataclass(frozen=True)
class Element:
    name: str        # lowercase SPICE name, e.g. "rd1__1_2"
    kind: str        # SPICE element letter
    nodes: tuple
    branch: bool     # ngspice keeps a branch current for it

    def __str__(self):
        return self.name


class Netlist:
    def __init__(self, title="ebana"):
        self.circuit = Circuit(title)
        self.circuit.raw(f".options TEMP={TEMPERATURE} TNOM={TEMPERATURE}")
        self._prefixes = set()

    def block(self, prefix):
        if prefix in self._prefixes:
            raise ValueError(f"block name '{prefix}' is already used")
        self._prefixes.add(prefix)
        return Block(self, prefix)

    def lines(self):
        return self.circuit.get_netlist()

    def __str__(self):
        return "\n".join(self.lines())


class Block:
    """Adds elements for one layer. Nodes are net names; values are numbers or {expressions}."""

    def __init__(self, netlist, prefix):
        self.netlist = netlist
        self.prefix = prefix

    def node(self, name):
        return f"{self.prefix}__{name}"

    def _id(self, id):
        return f"{self.prefix}__{id}"

    def _handle(self, kind, id, nodes, branch=False):
        return Element(name=f"{kind}{self._id(id)}".lower(), kind=kind, nodes=tuple(nodes), branch=branch)

    # passive <<<
    def resistor(self, id, a, b, resistance):
        self.netlist.circuit.R(self._id(id), (a, b), resistance)
        return self._handle("r", id, (a, b))
    # >>>

    # sources <<<
    def voltage_source(self, id, plus, minus, value):
        self.netlist.circuit.V(self._id(id), (plus, minus), value)
        return self._handle("v", id, (plus, minus), branch=True)

    def current_source(self, id, plus, minus, value):
        # the current flows from plus through the source to minus
        self.netlist.circuit.I(self._id(id), (plus, minus), value)
        return self._handle("i", id, (plus, minus))

    def behavioral_current(self, id, plus, minus, expression):
        self.netlist.circuit.B(self._id(id), (plus, minus), current_expression=expression)
        return self._handle("b", id, (plus, minus))

    def behavioral_voltage(self, id, plus, minus, expression):
        self.netlist.circuit.B(self._id(id), (plus, minus), voltage_expression=expression)
        return self._handle("b", id, (plus, minus), branch=True)
    # >>>

    # diodes <<<
    def diode(self, id, anode, cathode, model):
        self.netlist.circuit.D(self._id(id), (anode, cathode), model=model)
        return self._handle("d", id, (anode, cathode))
    # >>>

    # includes <<<
    def include(self, path):
        self.netlist.circuit.include(path)
    # >>>

    @staticmethod
    def array(elements, shape):
        array = np.empty(len(elements), dtype=object)
        array[:] = elements
        return array.reshape(shape)
