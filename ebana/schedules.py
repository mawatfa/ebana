"""Learning rate schedules, as in keras. They map the optimizer step (from 0) to a learning rate."""

import math
from abc import ABC, abstractmethod
from bisect import bisect_left


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, step):
        pass


def as_schedule(lr):
    return lr if isinstance(lr, LearningRateSchedule) else _Constant(lr)


class _Constant(LearningRateSchedule):
    def __init__(self, learning_rate):
        self.learning_rate = float(learning_rate)

    def __call__(self, step):
        return self.learning_rate


class ExponentialDecay(LearningRateSchedule):
    """lr = initial * decay_rate ** (step / decay_steps)"""

    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate * self.decay_rate**p


class InverseTimeDecay(LearningRateSchedule):
    """lr = initial / (1 + decay_rate * step / decay_steps)"""

    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def __call__(self, step):
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate / (1 + self.decay_rate * p)


class PiecewiseConstantDecay(LearningRateSchedule):
    """values[i] while boundaries[i - 1] < step <= boundaries[i]."""

    def __init__(self, boundaries, values):
        if len(values) != len(boundaries) + 1:
            raise ValueError("values needs one more entry than boundaries")
        self.boundaries = list(boundaries)
        self.values = list(values)

    def __call__(self, step):
        return self.values[bisect_left(self.boundaries, step)]


class PolynomialDecay(LearningRateSchedule):
    """From the initial to the end learning rate as (1 - step / decay_steps)^power."""

    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate=0.0, power=1.0, cycle=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def __call__(self, step):
        if self.cycle:
            decay_steps = self.decay_steps * max(1, math.ceil(step / self.decay_steps))
        else:
            decay_steps = self.decay_steps
            step = min(step, decay_steps)
        fraction = (1 - step / decay_steps) ** self.power
        return (self.initial_learning_rate - self.end_learning_rate) * fraction + self.end_learning_rate


class CosineDecay(LearningRateSchedule):
    """
    Cosine decay to alpha times the peak over decay_steps, after an optional
    linear warmup to warmup_target.
    """

    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warmup_steps=0, warmup_target=None):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.initial_learning_rate + (self.warmup_target - self.initial_learning_rate) * step / self.warmup_steps
        peak = self.warmup_target if self.warmup_steps else self.initial_learning_rate
        step = min(step - self.warmup_steps, self.decay_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        return peak * ((1 - self.alpha) * cosine + self.alpha)


class CosineDecayRestarts(LearningRateSchedule):
    """SGDR: every period is t_mul times longer and starts at m_mul times the previous peak."""

    def __init__(self, initial_learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0):
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha

    def __call__(self, step):
        completed = step / self.first_decay_steps
        if self.t_mul == 1.0:
            i_restart = math.floor(completed)
            completed -= i_restart
        else:
            i_restart = math.floor(math.log(1 - completed * (1 - self.t_mul), self.t_mul))
            sum_r = (1 - self.t_mul**i_restart) / (1 - self.t_mul)
            completed = (completed - sum_r) / self.t_mul**i_restart
        cosine = 0.5 * (1 + math.cos(math.pi * completed))
        return self.initial_learning_rate * self.m_mul**i_restart * ((1 - self.alpha) * cosine + self.alpha)
