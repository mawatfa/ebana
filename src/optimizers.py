#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
import pickle

#######################################################################
#                          optimizer methods                          #
#######################################################################

class SGD():
    def __init__(self, model, scale_factor=0.001):
        self.model = model
        self.scale_factor = scale_factor

    def step(self, X):
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_kind == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size
                    new_weights = layer.w - layer.lr / self.scale_factor * update
                    layer.w = np.clip(new_weights, 0.0000001, 1)

class SGDMomentum():
    def __init__(self, model, scale_factor=0.001, momentum=0.9):
        self.model = model
        self.momentum = momentum
        self.scale_factor = scale_factor
        self.v = {}
        for layer in self.model.computation_graph:
            if layer.grad:
                self.v[layer.name] = np.zeros(layer.shape)

    def step(self, X):
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_kind == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size
                    self.v[layer.name] = self.momentum * self.v[layer.name] + (1 - self.momentum) * update
                    new_weights = layer.w - layer.lr / self.scale_factor * self.v[layer.name]
                    #self.v[layer.name] = self.momentum * self.v[layer.name] - layer.lr / self.scale_factor * update
                    #new_weights = layer.w + self.v[layer.name]
                    layer.w = np.clip(new_weights, 0.0000001, 1)

class Adam():
    def __init__(self, model, scale_factor=0.001, beta1=0.5, beta2=0.5, eps=1e-8):
        self.model = model
        self.scale_factor = scale_factor
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 0
        self.c, self.v, self.s= {}, {}, {}
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_kind == 'dense_layer':
                    self.v[layer.name] = np.zeros(layer.shape)
                    self.s[layer.name] = np.zeros(layer.shape)
                elif layer.layer_kind in ['diode_layer', 'bias_voltage_layer']:
                    self.c[layer.name] = np.zeros(layer.shape)

    def step(self, X):
        self.k += 1
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_kind == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size

                    self.v[layer.name] = self.beta1 * self.v[layer.name] + (1 - self.beta1) * update
                    self.v[layer.name] = self.v[layer.name] / (1 - self.beta1 ** self.k)

                    self.s[layer.name] = self.beta2 * self.s[layer.name] + (1 - self.beta2) * (update ** 2)
                    self.s[layer.name] = self.s[layer.name] / (1 - self.beta2 ** self.k)

                    new_weights = layer.w - layer.lr * self.v[layer.name] / (np.sqrt(self.s[layer.name]) + self.eps)
                    layer.w = np.clip(new_weights, 0.0000001, 1)

                elif layer.layer_kind in ['diode_layer', 'bias_voltage_layer']:
                    #layer.bias_voltage -=  10 * layer.voltage_drops
                    #print(layer.bias_voltage)
                    pass

    def save_state(self, filepath):
        optimizer_state = {}
        for layer in self.model.computation_graph:
            if layer.layer_kind == 'dense_layer':
                optimizer_state[layer.name] = { 'weights' : layer.w,
                                                'velocity' : self.v,
                                                'acceleration' : self.s
                                                }
        with open(filepath, 'wb') as handle:
            pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath):

        with open(filepath, 'rb') as handle:
            optimizer_state = pickle.load(handle)

        for layer in self.model.computation_graph:
            if layer.layer_kind == 'dense_layer':
                layer.w = optimizer_state[layer.name]['weights']
                layer.v = optimizer_state[layer.name]['velocity']
                layer.s = optimizer_state[layer.name]['acceleration']
