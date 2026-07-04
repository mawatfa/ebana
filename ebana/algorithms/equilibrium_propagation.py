from .base import LearningAlgorithm, StepResult, free_losses, scaled


class EquilibriumPropagation(LearningAlgorithm):
    """
    Scellier and Bengio (2017). The outputs are nudged with -beta * dL/dy and
    the gradient is (dE/dp(nudged) - dE/dp(free)) / beta.
    """

    phases = ("free", "nudge")

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

    phases = ("free", "positive", "negative")

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
