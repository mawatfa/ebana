from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def __call__(self, y, target):
        """The loss of one sample and its gradient with respect to y."""


class MSE(Loss):
    def __call__(self, y, target):
        diff = y - target
        return 0.5 * np.sum(diff**2), diff


def softmax(y):
    e = np.exp(y - np.max(y))
    return e / np.sum(e)


def log_softmax(y):
    z = y - np.max(y)
    return z - np.log(np.sum(np.exp(z)))


def sigmoid(y):
    e = np.exp(-np.abs(y))
    return np.where(y >= 0, 1 / (1 + e), e / (1 + e))


class CrossEntropy(Loss):
    """Softmax cross-entropy of the outputs."""

    def __call__(self, y, target):
        return -np.sum(target * log_softmax(y)), softmax(y) - target


class BinaryCrossEntropy(Loss):
    """Sigmoid cross-entropy of each output."""

    def __call__(self, y, target):
        loss = np.sum(np.maximum(y, 0) - y * target + np.log1p(np.exp(-np.abs(y))))
        return loss, sigmoid(y) - target
