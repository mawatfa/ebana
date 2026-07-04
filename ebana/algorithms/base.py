from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class SampleCircuit:
    """The circuit loaded for one sample, as a learning algorithm uses it."""

    def __init__(self, model, sim, on_solve=None):
        self.model = model
        self.sim = sim
        self.on_solve = on_solve    # on_solve(phase, sim), called after every named solve

    @property
    def parameters(self):
        return self.model.trainable_parameters()

    def solve(self, phase=None):
        self.sim.solve()
        if phase is not None and self.on_solve is not None:
            self.on_solve(phase, self.sim)

    def outputs(self):
        return {layer.name: layer.read(self.sim) for layer in self.model.outputs}

    def measure(self):
        return {p.full_name: p.measure(self.sim) for p in self.parameters}

    def set(self, parameter, value):
        """Change the parameter's elements in the loaded circuit (not the parameter's value)."""
        parameter.device.alter(self.sim, parameter.elements, value)

    def nudge(self, forces):
        """Push each output with its force in amperes."""
        for layer in self.model.outputs:
            self.sim.alter(*layer.nudge(forces[layer.name]))


@dataclass
class StepResult:
    grads: dict         # parameter name -> gradient
    losses: dict        # output name -> loss
    outputs: dict       # output name -> free phase output
    forces: dict = field(default_factory=dict)


class LearningAlgorithm(ABC):
    phases = ("free",)

    @abstractmethod
    def train_step(self, circuit, targets, loss):
        """Return the StepResult of one sample. loss maps output names to losses."""

    def predict_step(self, circuit):
        return free_outputs(circuit)


def free_outputs(circuit):
    circuit.solve()
    return circuit.outputs()


def free_losses(outputs, targets, loss):
    values, grads = {}, {}
    for name, y in outputs.items():
        values[name], grads[name] = loss[name](y, targets[name])
    return values, grads


def scaled(forces, factor):
    return {name: factor * np.asarray(f) for name, f in forces.items()}
