#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                    methods for calculating loss                     #
#######################################################################

class MSE:
    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Mean Squared Error (MSE)
        """
        prediction = output_node_voltages[:,0] - output_node_voltages[:,1]

        if mode == 'train':
            num_output_nodes = output_node_voltages.shape[0]
            losses = np.zeros(shape=(num_output_nodes, ))
            currents = np.zeros(shape=(num_output_nodes, 2))

            # MSE calculatuon
            diff = prediction - target
            losses = 0.5 * np.power(diff, 2)

            # loss current calculation
            #beta = np.random.choice([-1, 1]) * beta
            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prediction.reshape(1, -1)

    def verify_result(self, target, prediction):
        c = np.product(np.equal(target, np.round(prediction, 0)), axis=1)
        return c

class BCE:
    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Binary Cross Entropy Loss (BCE)
        """
        prediction = output_node_voltages[:,0] - output_node_voltages[:,1]
        prob = 1 / (1 + np.exp(-prediction)) # sigmoid function

        if mode == 'train':
            eps = 0.0000001
            losses = np.zeros(shape=(1, ))
            currents = np.zeros(shape=(1, 2))

            losses = - ((target * np.log10(prob + eps) + (1 - target) * np.log(1 - prob + eps)))
            diff = prob - target

            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prob

    def verify_result(self, target, prediction):
        c = np.product(np.equal(target, np.round(prediction, 0)), axis=1)
        return c

class CrossEntropyLoss:
    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Cross Entropy Loss
        """
        prediction = output_node_voltages[:,0] - output_node_voltages[:,1]
        prob = np.exp(prediction) / np.sum(np.exp(prediction))

        if mode == 'train':
            eps = 0.0000001
            output_nodes = output_node_voltages.shape[0]
            losses = np.zeros(shape=(output_nodes, ))
            currents = np.zeros(shape=(output_nodes, 2))

            # cross-entropy loss calculation
            losses = - target * np.log(prob + eps)
            diff = prob - target

            # loss current calculation
            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prob

    def verify_result(self, target, prediction):
        a = np.argmax(prediction, axis=1)
        b = np.zeros((a.size, a.max()+1), dtype=int)
        b[np.arange(a.size),a] = 1
        c = np.product(np.equal(target, b), axis=1)
        return c
