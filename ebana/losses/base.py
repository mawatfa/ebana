import numpy as np
from abc import ABC, abstractmethod
from .utils import default_loss_update_rule


class BaseLoss(ABC):
    def __init__(self, beta: float):
        """
        Base class for loss functions.
        """
        self.beta = beta
        self.set_loss_update_equation(func=None)

    def set_loss_update_equation(self, func):
        """
        Set the function to update loss. Use the default rule if none provided.
        """
        self.loss_update_rule = func if func else default_loss_update_rule

    def compute_output(self, prediction: np.ndarray):
        """
        Processes prediction before applying it to the loss function.
        To be implemented in subclasses.
        """
        pass

    @abstractmethod
    def verify_result(self, prediction: np.ndarray, target: np.ndarray):
        """
        Verifies whether predictions match the target.
        To be implemented in the subclasses for specific verification logic.
        """
        pass

    @abstractmethod
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass
