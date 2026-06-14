#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                    weight initialization methods                    #
#######################################################################

class Initializers:
    def __init__(self, init_type, params=None):
        self.init_type = init_type
        self.params = params
        self.set_bounds()

    def set_bounds(self):
        if self.params:
            self.init_low = self.params["init_low"]
            self.init_high = self.params["init_high"]
            self.clip_low = self.params["clip_low"]
            self.clip_high = self.params["clip_high"]
            assert self.init_low > 0 and self.init_high > 0 and self.clip_low and self.clip_high
        else:
            self.init_low = 1e-7
            self.init_high = 1e-5
            self.clip_low = self.init_low
            self.clip_high = self.init_high

    def clip_conductances(self, w):
        return np.clip(w, self.clip_low, self.clip_high)

    def initialize_weights(self, shape):
        if self.init_type == 'random_uniform':
            return self.random_uniform(shape)
        elif self.init_type == 'log_uniform':
            return self.log_uniform(shape)
        elif self.init_type == 'glorot_uniform':
            return self.glorot_uniform(shape)
        elif self.init_type == 'glorot_log_uniform':
            return self.glorot_log_uniform(shape)
        elif self.init_type == 'he_uniform':
            return self.he_uniform(shape)

    def random_uniform(self, shape):
        return np.random.uniform(self.init_low, self.init_high, size=shape)

    def log_uniform(self, shape):
        w = np.exp(np.random.uniform(np.log(self.init_low), np.log(self.init_high), size=shape))
        return np.clip(w, self.init_low, self.init_high)

    def glorot_uniform(self, shape):
        fan_in, fan_out = shape
        limit = self.init_high / np.sqrt(fan_in + fan_out + 1)
        w = np.random.uniform(self.init_low, limit, size=shape)
        return np.clip(w, self.init_low, self.init_high)

    def glorot_log_uniform(self, shape):
        fan_in, fan_out = shape
        limit = np.log10(self.init_high) / np.sqrt(fan_in + fan_out + 1)
        w = 10**(np.random.uniform(np.log10(self.init_low), limit, size=shape))
        w = np.clip(w, self.init_low, self.init_high)
        return w

    def he_uniform(self, shape):
        fan_in = shape[0]
        limit = self.init_high / np.sqrt(fan_in + 1)
        w = np.random.uniform(self.init_low, limit, size=shape)
        return np.clip(w, self.init_low, self.init_high)
