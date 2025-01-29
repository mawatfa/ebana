import numpy as np
from abc import abstractmethod
from .base import BaseLoss
from .utils import calculate_prediction, verify_result_using_probability

class MSEBase(BaseLoss):
    def __init__(self, beta):
        super().__init__(beta)

    @abstractmethod
    def verify_result(self, target, prediction):
        pass

    def __call__(self, prediction, target):
        """
        Calculate losses and gradient currents for MSE.
        """
        diff = prediction - target
        losses = 0.5 * np.square(diff)  # Vectorized loss calculation

        # Update current injections using loss update rule
        currents = - self.loss_update_rule(self.beta, diff)

        # print(f"Prediction: {np.round(prediction, 2)}, Target: {target}")
        return losses, currents


class MSE(MSEBase):
    def __init__(self, beta, output_midpoint=0.0):
        self.output_midpoint = output_midpoint
        super().__init__(beta)

    def verify_result(self, prediction: np.ndarray, target: np.ndarray):
        """
        Verifies whether predictions match the target using binary comparison.
        :param target: The expected output.
        :param prediction: The actual prediction.
        """
        # Convert target and prediction into binary arrays based on output midpoint
        target_arr = target >= self.output_midpoint
        pred_arr = prediction >= self.output_midpoint

        # If prediction is 1-dimensional (single sample)
        if prediction.ndim == 1:
            # Check if the single target and prediction match
            mask = np.any(target_arr != pred_arr)  # Find mismatches in 1D case
            return int(np.logical_not(mask))  # Return 1 if match, 0 if not

        # Multi-sample case (2D input)
        else:
            # Check for mismatches across all samples (rows)
            mask = np.any( target_arr != pred_arr, axis=1)  # Find mismatches for each sample
            return np.logical_not(mask).astype(int)  # Return array of matches (1 if match, 0 if not)


class MSEProb(MSEBase):
    def __init__(self, beta):
        super().__init__(beta)

    def verify_result(self, prediction: np.ndarray, target: np.ndarray):
        """
        Verifies whether predictions match the target using binary comparison.
        :param target: The expected output.
        :param prediction: The actual prediction.
        """
        return verify_result_using_probability(target, prediction)
