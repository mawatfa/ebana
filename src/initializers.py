#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                    weight initialization methods                    #
#######################################################################

class Initializers:
    def __init__(self, init_type, params):
        self.init_type = init_type
        self.params = params

    def get_bounds(self):
        if self.params["L"]:
            L = self.params["L"]
        else:
            L = self.defaults["L"]
        if self.params["U"]:
            U = self.params["U"]
        else:
            U = self.defaults["U"]
        return L, U

    def clip_conductances(self, w):
        if self.params["g_min"]:
            w = np.clip(w, self.params["g_min"], None)
        if self.params["g_max"]:
            w = np.clip(w, None, self.params["g_max"])
        return w

    def initialize_weights(self, shape):
        if self.init_type == 'random_uniform':
            return self.random_uniform(shape)
        elif self.init_type == 'glorot':
            return self.glorot(shape)

    def random_uniform(self, shape):
        self.defaults = {"L": 1e-4, "U": 1e-1}
        L, U = self.get_bounds()
        return np.random.uniform(L, U, size=shape)

    def glorot(self, shape):
        self.defaults = {"L": 1e-7, "U": 8e-6}
        L, U = self.get_bounds()
        return np.random.uniform(L, U / np.sqrt(shape[0] + shape[1] + 1), size=shape)
        # a = np.exp(np.random.uniform(np.log(L), np.log(U) / np.sqrt(shape[0] + shape[1] + 1), size=shape))
        # return a # log uniform
