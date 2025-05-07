import numpy as np

class LearningRateSchedule:
    def __call__(self, step: int) -> float:
        raise NotImplementedError

class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, step: int) -> float:
        return self.learning_rate

class CosineDecay(LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        alpha: float = 0.0,
    ):
        """
        Args:
            initial_learning_rate: The initial learning rate.
            decay_steps: Number of steps to decay over.
            alpha: Minimum learning rate value for decay as a fraction of `initial_learning_rate`.
        """
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be > 0, got {decay_steps}")
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step: int) -> float:
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed
