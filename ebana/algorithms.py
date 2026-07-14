from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class SampleCircuit:
    """The circuit loaded for one sample, as a learning algorithm uses it."""

    def __init__(self, model, sim, on_solve=None):
        self.model = model
        self.sim = sim
        self.on_solve = on_solve    # on_solve(phase, sim), called after every named solve
        self._set, self._nudged = {}, {}

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
        self._set[parameter.full_name] = parameter

    def nudge(self, forces):
        """Push each output with its force in amperes."""
        for layer in self.model.outputs:
            self.sim.alter(*layer.nudge(forces[layer.name]))
            self._nudged[layer.name] = layer

    def restore(self):
        """Undo set() and nudge() before the next sample."""
        for p in self._set.values():
            p.device.alter(self.sim, p.elements, p.value)
        for layer in self._nudged.values():
            self.sim.alter(*layer.nudge(np.zeros(layer.units)))
        self._set, self._nudged = {}, {}


@dataclass
class StepResult:
    grads: dict         # parameter name -> gradient
    losses: dict        # output name -> loss
    outputs: dict       # output name -> free phase output
    forces: dict = field(default_factory=dict)


class LearningAlgorithm(ABC):
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


class EquilibriumPropagation(LearningAlgorithm):
    """
    Scellier and Bengio (2017). The outputs are nudged with -beta * dL/dy and
    the gradient is (dE/dp(nudged) - dE/dp(free)) / beta.
    """

    def __init__(self, beta):
        if beta == 0:
            raise ValueError("beta must be non-zero")
        self.beta = beta

    def train_step(self, circuit, targets, loss):
        circuit.solve("free")
        outputs = circuit.outputs()
        free = circuit.measure()
        losses, grads = free_losses(outputs, targets, loss)

        forces = scaled(grads, -self.beta)
        circuit.nudge(forces)
        circuit.solve("nudge")
        nudged = circuit.measure()

        return StepResult(
            grads={p.full_name: p.gradient(free[p.full_name], nudged[p.full_name]) / self.beta
                   for p in circuit.parameters},
            losses=losses, outputs=outputs, forces=forces,
        )


class SymmetricEquilibriumPropagation(EquilibriumPropagation):
    """
    Laborieux et al. (2021). Nudges with +beta and -beta and uses
    (dE/dp(+beta) - dE/dp(-beta)) / (2 beta), which removes the first order
    error of the one-sided estimate.
    """

    def train_step(self, circuit, targets, loss):
        circuit.solve("free")
        outputs = circuit.outputs()
        losses, grads = free_losses(outputs, targets, loss)

        forces = scaled(grads, -self.beta)
        circuit.nudge(forces)
        circuit.solve("positive")
        positive = circuit.measure()
        circuit.nudge(scaled(forces, -1))
        circuit.solve("negative")
        negative = circuit.measure()

        return StepResult(
            grads={p.full_name: p.gradient(negative[p.full_name], positive[p.full_name]) / (2 * self.beta)
                   for p in circuit.parameters},
            losses=losses, outputs=outputs, forces=forces,
        )
