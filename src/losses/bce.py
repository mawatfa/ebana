import numpy as np
from .base import BaseLoss
from .utils import calculate_prediction, verify_result_using_probability

class BCE(BaseLoss):
    def __init__(self, beta, fold=False):
        """
        Initialize BCE loss class.
        """
        super().__init__(beta, fold)

    def __call__(self, output_node_voltages, target=None, mode='training'):
        """
        Calculate losses and gradient currents for BCE.
        """
        prediction = calculate_prediction(output_node_voltages, self.fold)
        prob = 1 / (1 + np.exp(-prediction))

        if mode == 'evaluating':
            return prob.reshape(1, -1)
        elif mode == 'predicting':
            return output_node_voltages, prob.reshape(1, -1)
        elif mode == 'training':
            eps = np.finfo(float).eps
            losses = -(target * np.log(prob + eps) + (1 - target) * np.log(1 - prob + eps))
            diff = prob - target
            currents = self.loss_update_rule(self.beta, diff)
            return losses, currents

    def verify_result(self, target, prediction):
        """
        Verifies whether predictions match the target using probability comparison.
        """
        return verify_result_using_probability(target, prediction)
