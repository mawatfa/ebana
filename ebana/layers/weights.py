import math

import numpy as np

from ..constraints import MaxRelativeChange
from ..devices import Resistor
from ..simulation import GROUND
from .base import Layer, net_names


class Dense(Layer):
    """A crossbar from every input net to every output net. weight has shape (inputs, units)."""

    def __init__(self, name, units, initializer=None, device=None, lr=0.0, trainable=True,
                 constraint=MaxRelativeChange(0.5), grad_func=None, update_func=None):
        super().__init__(name)
        self.units = units
        self.initializer = initializer
        self.device = device or Resistor()
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable,
                            grad_func=grad_func, update_func=update_func)

    def build(self, inputs):
        (x,) = inputs
        if x.ndim != 1:
            raise ValueError(f"Dense '{self.name}' needs a 1-D input, got shape {x.shape}")
        self.inputs = x
        self.weight = self.add_parameter("weight", self.initializer((len(x), self.units)), self.device, **self.options)
        return net_names((self.units,), f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        W = self.weight.value
        elements = np.empty(W.shape, dtype=object)
        for i in range(W.shape[1]):
            for j in range(W.shape[0]):
                elements[j, i] = self.device.stamp(block, f"{j + 1}_{i + 1}", self.inputs[j], self.output[i], W[j, i])
        self.weight.elements = elements


def _padding(size, kernel, stride, padding):
    if padding == "valid":
        return 0, 0
    total = max((math.ceil(size / stride) - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


def _patches(x, kernel_size, stride, padding):
    """Input nets under every kernel position, shape (out_h, out_w, kh, kw, channels). Padding is ground."""
    (kh, kw), (sh, sw) = kernel_size, stride
    pad_h, pad_w = _padding(x.shape[0], kh, sh, padding), _padding(x.shape[1], kw, sw, padding)
    padded = np.pad(x, (pad_h, pad_w, (0, 0)), mode="constant", constant_values=GROUND)
    out_h = (padded.shape[0] - kh) // sh + 1
    out_w = (padded.shape[1] - kw) // sw + 1
    patches = np.empty((out_h, out_w, kh, kw, x.shape[2]), dtype=object)
    for oh in range(out_h):
        for ow in range(out_w):
            patches[oh, ow] = padded[oh * sh: oh * sh + kh, ow * sw: ow * sw + kw, :]
    return patches


class LocallyConnected2D(Layer):
    """
    A 2-D convolution without weight sharing: every output position has its
    own crossbar from its patch to the filters. weight has shape
    (out_h, out_w, kh * kw * channels, filters).
    """

    def __init__(self, name, kernel_size, stride, filters, padding="valid", initializer=None, device=None,
                 lr=0.0, trainable=True, constraint=MaxRelativeChange(0.5), grad_func=None,
                 update_func=None):
        super().__init__(name)
        self.kernel_size, self.stride, self.filters, self.padding = tuple(kernel_size), tuple(stride), filters, padding
        self.initializer = initializer
        self.device = device or Resistor()
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable,
                            grad_func=grad_func, update_func=update_func)

    def build(self, inputs):
        (x,) = inputs
        patches = _patches(x, self.kernel_size, self.stride, self.padding)
        out_h, out_w = patches.shape[:2]
        self.patch_inputs = patches.reshape(out_h, out_w, -1)
        k = self.patch_inputs.shape[-1]
        # initialize each patch on its own, with its own fan-in and fan-out
        W = np.empty((out_h, out_w, k, self.filters))
        for oh in range(out_h):
            for ow in range(out_w):
                W[oh, ow] = self.initializer((k, self.filters))
        self.weight = self.add_parameter("weight", W, self.device, **self.options)
        return net_names((out_h, out_w, self.filters), f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        W = self.weight.value
        elements = np.empty(W.shape, dtype=object)
        out_h, out_w, k, filters = W.shape
        for oh in range(out_h):
            for ow in range(out_w):
                rows, cols = self.patch_inputs[oh, ow], self.output[oh, ow]
                for i in range(filters):
                    for j in range(k):
                        elements[oh, ow, j, i] = self.device.stamp(
                            block, f"{oh + 1}_{ow + 1}_{j + 1}_{i + 1}", rows[j], cols[i], W[oh, ow, j, i])
        self.weight.elements = elements


class AveragePooling2D(Layer):
    """
    Each output is tied to its window through equal conductances, so it sits
    at the window's average voltage when it draws no current. weight has
    shape (out_h, out_w, kh * kw, channels) and is not trained by default.
    """

    def __init__(self, name, pool_size=(2, 2), strides=None, padding="valid", initializer=None,
                 device=None, lr=0.0, trainable=False, constraint=None):
        super().__init__(name)
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides) if strides is not None else self.pool_size
        self.padding = padding
        self.initializer = initializer
        self.device = device or Resistor()
        self.options = dict(lr=lr, constraint=constraint, trainable=trainable)

    def build(self, inputs):
        (x,) = inputs
        patches = _patches(x, self.pool_size, self.strides, self.padding)
        out_h, out_w, kh, kw, channels = patches.shape
        self.windows = patches.reshape(out_h, out_w, kh * kw, channels)
        self.weight = self.add_parameter("weight", self.initializer(self.windows.shape), self.device, **self.options)
        return net_names((out_h, out_w, channels), f"n_{self.name}")

    def stamp(self, block, data, sample_index):
        W = self.weight.value
        elements = np.empty(W.shape, dtype=object)
        for (oh, ow, j, c), net in np.ndenumerate(self.windows):
            elements[oh, ow, j, c] = self.device.stamp(block, f"{oh + 1}_{ow + 1}_{j + 1}_{c + 1}", net,
                                                       self.output[oh, ow, c], W[oh, ow, j, c])
        self.weight.elements = elements
