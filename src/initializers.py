#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                    weight initialization methods                    #
#######################################################################

class Initializers:
    def __init__(self, shape):
        self.shape = shape

    def random_uniform(self, L=0.0001, U=0.1):
        return np.random.uniform(L, U, size=self.shape)

    def glorot(self, L=0.0000001, U=0.08):
        return np.random.uniform(L, U / np.sqrt(self.shape[0] + self.shape[1] + 1), size=self.shape)
