import numpy as np
from .base import BaseLoss
from .utils import calculate_prediction, verify_result_using_probability

def sparsemax(logits_):
    """
    Sparsemax function for matrices: Converts logits into a sparse probability distribution.
    :param logits: 2D array of unnormalized logits (scores) as input.
    :return: Sparse probability distribution for each row of logits.
    """
    if logits_.ndim == 1:
        logits = logits_.reshape(1, -1)
    else:
        logits = logits_

    # Sort logits along the last axis (per row)
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    cumulative_sum = np.cumsum(sorted_logits, axis=1)
    k = np.arange(1, logits.shape[1] + 1)

    # Calculate the threshold (rho) for each row
    condition = sorted_logits - (cumulative_sum - 1) / k > 0
    rho = np.sum(condition, axis=1)  # rho is the number of positive entries in each row

    # Calculate tau for each row
    tau = (cumulative_sum[np.arange(logits.shape[0]), rho - 1] - 1) / rho

    # Apply sparsemax formula to get the sparse probabilities
    sparse_probs = np.maximum(0, logits - tau[:, np.newaxis])

    if logits_.ndim == 1:
        return tau[0], sparse_probs[0]
    else:
        return tau, sparse_probs

def sparsemax_loss(prediction_, target_, sparse_prob_, tau_):
    """
    Calculate sparsemax loss.
    :param prediction: 2D array of logits (unnormalized scores).
    :param target: 2D array of target values (can be any two distinct numbers).
    :param sparse_prob: Sparsemax probabilities.
    :param tau: Sparsemax threshold value for each row.
    :return: Sparsemax loss for each row.
    """
    if prediction_.ndim == 1:
        prediction = prediction_.reshape(1, -1)
        target = target_.reshape(1, -1)
        sparse_prob = sparse_prob_.reshape(1, -1)
        tau = np.array([tau_])
    else:
        prediction = prediction_
        target = target_
        sparse_prob = sparse_prob_
        tau = tau_

    if target_.ndim == 1:
        target = target_.reshape(1, -1)

    # Normalize the target values to 0 and 1
    max_target = np.max(target, axis=1, keepdims=True)

    # Convert target to 0 and 1
    target_binary = (target == max_target).astype(float)

    # Logit corresponding to the true label (z_y) for each row
    z_y = np.sum(prediction * target_binary, axis=1)

    # Compute the loss using the sparsemax loss formula
    support_set = sparse_prob > 0  # Identify the support set where sparsemax > 0
    sparse_loss = 0.5 - z_y + 0.5 * np.sum(support_set * (prediction**2 - tau[:, np.newaxis]**2), axis=1)

    return sparse_loss[0] if prediction_.ndim == 1 else sparse_loss

class SparsemaxLoss(BaseLoss):
    def __init__(self, beta, fold=False):
        """
        Initialize Sparsemax loss class.
        :param beta: Scaling factor for loss updates.
        """
        super().__init__(beta, fold)

    def __call__(self, output_node_voltages, target=None, mode='training'):
        """
        Calculate losses and gradient currents for sparsemax loss.
        :param output_node_voltages: Voltage values at the output nodes.
        :param target: Target values for training.
        :param mode: Specifies the operation mode ('training', 'evaluating', 'predicting').
        """
        prediction = calculate_prediction(output_node_voltages, self.fold)

        # Apply Sparsemax
        tau, prob = sparsemax(prediction)

        if mode == 'evaluating':
            return prob.reshape(1, -1)

        elif mode == 'predicting':
            return output_node_voltages, prob.reshape(1, -1)

        elif mode == 'training':
            # Calculate sparsemax loss
            losses = np.zeros_like(prediction)
            losses[np.argmax(target)] = sparsemax_loss(prediction, target, prob, tau)

            # Calculate sparsemax gradient
            diff = prob - target
            currents = - self.loss_update_rule(self.beta, diff)

            return losses, currents

    def verify_result(self, target, prediction):
        """
        Verifies whether predictions match the target by selecting the max probability.
        :param target: The expected output.
        :param prediction: The actual prediction.
        """
        return verify_result_using_probability(target, prediction)
