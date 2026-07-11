"""
A layer is connected like y = Dense(...)(x) and implements:

  build(inputs)                   create the parameters and return the output nets
  stamp(block, data, index)       add the elements to the netlist and store the
                                  handles in parameter.elements
  set_sample(sim, data, index)    only for layers that depend on the data, like
                                  Input: alter the loaded circuit for another sample
"""

import itertools
from abc import ABC, abstractmethod

import numpy as np

from ..constraints import as_constraint
from ..schedules import as_schedule


def net_names(shape, prefix):
    names = np.empty(shape, dtype=object)
    for idx in np.ndindex(*shape):
        names[idx] = prefix + "".join(f"_{i + 1}" for i in idx)
    return names


class Parameter:
    """
    A value per element of a layer (conductances, bias voltages, gains).

    grad_func    f(a, b) used instead of the device's gradient
    update_func  f(parameter, update, step) returning the new value, used
                 instead of constraint(value - lr(step) * update)
    """

    def __init__(self, layer, name, value, device, lr=0.0, constraint=None,
                 trainable=False, grad_func=None, update_func=None):
        self.layer = layer
        self.name = name
        self.value = np.asarray(value, dtype=float)
        self.device = device
        self.lr = as_schedule(lr)
        self.constraint = as_constraint(constraint)
        self.trainable = trainable
        self.grad_func = grad_func
        self.update_func = update_func
        self.elements = None
        self.grad = np.zeros_like(self.value)

    @property
    def full_name(self):
        return f"{self.layer.name}/{self.name}"

    @property
    def shape(self):
        return self.value.shape

    def measure(self, sim):
        return self.device.measure(sim, self.elements)

    def gradient(self, a, b):
        if self.grad_func is not None:
            return self.grad_func(a, b)
        return self.device.gradient(a, b)

    def apply_update(self, update, step):
        if self.update_func is not None:
            self.value = np.asarray(self.update_func(self, update, step), dtype=float)
        else:
            self.value = self.constraint(self.value - self.lr(step) * update, self.value)

    def __repr__(self):
        return f"Parameter({self.full_name}, shape={self.shape})"


# Layers are numbered when connected, so parents always come before their
# children. Creation order doesn't work: in Dense(...)(Concatenate(...)(x))
# the Dense is created first.
_connection_order = itertools.count()


class Layer(ABC):
    def __init__(self, name):
        self.index = None
        self.name = name.lower()
        self.parents = []
        self.children = []
        self.params = {}
        self.output = None

    def __call__(self, inputs):
        if self.output is not None:
            raise ValueError(f"layer '{self.name}' is already connected")
        self.parents = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        for parent in self.parents:
            parent.children.append(self)
        self.output = np.asarray(self.build([p.output for p in self.parents]), dtype=object)
        self.index = next(_connection_order)
        return self

    @property
    def output_shape(self):
        return self.output.shape

    def add_parameter(self, name, value, device, **options):
        self.params[name] = Parameter(self, name, value, device, **options)
        return self.params[name]

    @abstractmethod
    def build(self, inputs):
        pass

    @abstractmethod
    def stamp(self, block, data, sample_index):
        pass

    def set_sample(self, sim, data, sample_index):
        pass

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class SourceLayer(Layer):
    """A layer without parents, built when created."""

    def __init__(self, name):
        super().__init__(name)
        self.output = np.asarray(self.build([]), dtype=object)
        self.index = next(_connection_order)

    def __call__(self, inputs):
        raise TypeError(f"'{type(self).__name__}' layers have no inputs")
