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


@dataclass(frozen=True)
class Element:
    name: str        # lowercase SPICE name, e.g. "rd1__1_2"
    kind: str        # SPICE element letter
    nodes: tuple
    branch: bool     # ngspice keeps a branch current for it

    def __str__(self):
        return self.name


class Netlist:
    def __init__(self, title="ebana", temperature=27):
        self.circuit = Circuit(title)
        self.circuit.raw(f".options TEMP={temperature} TNOM={temperature}")
        self._prefixes = set()
        self._models = set()
        self._includes = set()
        self.veriloga_files = set()

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

    def capacitor(self, id, a, b, capacitance):
        self.netlist.circuit.C(self._id(id), (a, b), capacitance)
        return self._handle("c", id, (a, b))

    def inductor(self, id, a, b, inductance):
        self.netlist.circuit.L(self._id(id), (a, b), inductance)
        return self._handle("l", id, (a, b), branch=True)
    # >>>

    # sources <<<
    def voltage_source(self, id, plus, minus, value):
        self.netlist.circuit.V(self._id(id), (plus, minus), value)
        return self._handle("v", id, (plus, minus), branch=True)

    def current_source(self, id, plus, minus, value):
        # the current flows from plus through the source to minus
        self.netlist.circuit.I(self._id(id), (plus, minus), value)
        return self._handle("i", id, (plus, minus))

    def vcvs(self, id, plus, minus, control_plus, control_minus, gain):
        self.netlist.circuit.E(self._id(id), (plus, minus, control_plus, control_minus), str(gain))
        return self._handle("e", id, (plus, minus), branch=True)

    def vccs(self, id, plus, minus, control_plus, control_minus, transconductance):
        self.netlist.circuit.G(self._id(id), (plus, minus, control_plus, control_minus), str(transconductance))
        return self._handle("g", id, (plus, minus))

    def cccs(self, id, plus, minus, control, gain):
        self.netlist.circuit.F(self._id(id), (plus, minus), f"{control.name} {gain}")
        return self._handle("f", id, (plus, minus))

    def ccvs(self, id, plus, minus, control, transresistance):
        self.netlist.circuit.H(self._id(id), (plus, minus), f"{control.name} {transresistance}")
        return self._handle("h", id, (plus, minus), branch=True)

    def behavioral_current(self, id, plus, minus, expression):
        self.netlist.circuit.B(self._id(id), (plus, minus), current_expression=expression)
        return self._handle("b", id, (plus, minus))

    def behavioral_voltage(self, id, plus, minus, expression):
        self.netlist.circuit.B(self._id(id), (plus, minus), voltage_expression=expression)
        return self._handle("b", id, (plus, minus), branch=True)
    # >>>

    # semiconductors <<<
    def diode(self, id, anode, cathode, model, **params):
        self.netlist.circuit.D(self._id(id), (anode, cathode), model=model, **params)
        return self._handle("d", id, (anode, cathode))

    def mosfet(self, id, drain, gate, source, bulk, model, **params):
        self.netlist.circuit.M(self._id(id), (drain, gate, source, bulk), model=model, **params)
        return self._handle("m", id, (drain, gate, source, bulk))

    def bjt(self, id, collector, base, emitter, model):
        self.netlist.circuit.Q(self._id(id), (collector, base, emitter), model=model)
        return self._handle("q", id, (collector, base, emitter))

    def veriloga(self, id, nodes, module, path, **params):
        # the file is compiled with OpenVAF when the netlist is loaded
        self.netlist.veriloga_files.add(str(path))
        model = self._id(f"{module}_{id}")
        self.netlist.circuit.model(model, module, **params)
        self.netlist.circuit.N(self._id(id), tuple(nodes), model)
        return self._handle("n", id, tuple(nodes))
    # >>>

    # models and includes <<<
    def model(self, name, model_type, **params):
        """Add a .model card for this block (once) and return its SPICE name."""
        full = self._id(name)
        if full not in self.netlist._models:
            self.netlist._models.add(full)
            self.netlist.circuit.model(full, model_type, **params)
        return full

    def include(self, path):
        if path not in self.netlist._includes:
            self.netlist._includes.add(path)
            self.netlist.circuit.include(path)

    def raw(self, line):
        self.netlist.circuit.raw(line)
    # >>>

    @staticmethod
    def array(elements, shape):
        array = np.empty(len(elements), dtype=object)
        array[:] = elements
        return array.reshape(shape)
