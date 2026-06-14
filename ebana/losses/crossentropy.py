import numpy as np
from .base import BaseLoss
from .utils import verify_result_using_probability

class CrossEntropyLoss(BaseLoss):
    def __init__(self, beta):
        """
        Initialize Cross Entropy loss class.
        """
        super().__init__(beta)

    def verify_result(self, prediction: np.ndarray, target: np.ndarray):
        """
        Verifies whether predictions match the target by selecting the max probability.
        :param target: The expected output.
        :param prediction: The actual prediction.
        """
        return verify_result_using_probability(target, prediction)

    def __call__(self, prediction, target):
        """
        Calculate losses and gradient currents for cross-entropy loss.
        """
        eps = np.finfo(float).eps  # Machine epsilon to prevent log(0)
        losses = - target * np.log(prediction + eps)  # Cross-entropy loss calculation

        # Calculate cross-entropy gradient
        diff = prediction - target
        currents = - self.loss_update_rule(self.beta, diff)

        return losses, currents
