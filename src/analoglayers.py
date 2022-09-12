#######################################################################
#                               imports                               #
#######################################################################

import uuid
import numpy as np
from .netlistgenerator import VoltageSources, DenseResistorMatrix, Diode, MOSDiode, BidirectionalAmplifier, CurrentSources
from .initializers import Initializers

#######################################################################
#                              NN Layers                              #
#######################################################################

class BaseLayer:
    def __init__(self, units, name, layer_kind, trainable=False):
        self.units = units
        self.name = name.lower()
        self.layer_kind = layer_kind
        self.trainable = trainable
        self.built = False

        self.input_shape = None
        self.output_shape = None
        self.shape = None

        self.layer_id = uuid.uuid1()
        self.previous_layer = []
        self.next_layer = []

    def define_connections(self, inputs):
        self.previous_layer = inputs

        if isinstance(inputs, list):
            for previous_layer in inputs:
                previous_layer.next_layer.append(self)
        else:
            self.previous_layer.next_layer.append(self)

    def __call__(self, inputs, mode=None):
        if not self.built:
            self.build(inputs)
            self.built = True

        if mode == "executing":
            self.call(self, inputs)
        else:
            return self

class OnePort(BaseLayer):
    def __init__(self, units, name, layer_kind='one_port', trainable=False):
        super().__init__(units, name, layer_kind, trainable)

    def _generate_subcircuit_net_names(self):
        self.out_subcircuit_net_names = [f"P{i+1}" for i in range(self.units)]

    def _generate_net_names(self):
        self.out_net_names = [f"n_{self.name}_{i+1}" for i in range(self.units)]

class TwoPorts(BaseLayer):
    def __init__(self, units, name, layer_kind='two_ports', trainable=False):
        super().__init__(units, name, layer_kind, trainable)

    def _generate_subcircuit_net_names(self):
        self.in_subcircuit_net_names = [f"IN{i+1}" for i in range(self.input_shape[0])]
        self.out_subcircuit_net_names = [f"OUT{i+1}" for i in range(self.output_shape[0])]

    def _generate_net_names(self):
        self.in_net_names = self.previous_layer.out_net_names
        self.out_net_names = [f"n_{self.name}_{i+1}" for i in range(self.output_shape[0])]

class VoltageLayer(OnePort):
    def __init__(self, units, name, layer_kind='voltage_layer', trainable=False):
        super().__init__(units, name, layer_kind, trainable)

    def call(self, circuit, V):
        circuit.subcircuit(VoltageSources(self.name, self.out_subcircuit_net_names, V))
        circuit.X(self.name, self.name, *self.out_net_names)

    def _generate_subcircuit_current_net_names(self):
        self.current_net_names = [f"v.x{self.name}.v{i+1}" for i in range(self.units)]

class InputVoltageLayer(VoltageLayer):
    def __init__(self, units, name, trainable=False):
        super().__init__(units, name, layer_kind='input_voltage_layer', trainable=trainable)
        self.build()

    def build(self):
        self.built = True
        self.output_shape = (self.units, )
        self.shape = self.output_shape

        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

        if self.trainable:
            self.voltage_drops = np.zeros(shape=self.shape)
            super()._generate_subcircuit_current_net_names()

class BiasVoltageLayer(VoltageLayer):
    def __init__(self, units, name, bias_voltage, trainable=False):
        self.bias_voltage = bias_voltage
        super().__init__(units, name, layer_kind='bias_voltage_layer', trainable=trainable)
        self.build()

    def build(self):
        self.built = True
        self.output_shape = (self.units,)
        self.shape = self.output_shape

        super()._generate_subcircuit_net_names()
        super()._generate_net_names()
        if self.trainable:
            self.voltage_drops = np.zeros(shape=self.shape)
            super()._generate_subcircuit_current_net_names()

class DenseLayer(TwoPorts):
    def __init__(self, units, name, lr=0.001, init_type='random_uniform', trainable=True):
        super().__init__(units, name, layer_kind='dense_layer', trainable=trainable)
        self.lr = lr
        self.init_type = init_type

    def _initialize_variables(self):
        initializers = Initializers(shape=self.shape)
        if self.init_type == 'random_uniform':
            self.w = initializers.random_uniform()
        else:
            self.w = initializers.glorot()
        self.voltage_drops = np.zeros(shape=self.shape)

    def _set_layer_shapes(self):
        self.input_shape = self.previous_layer.output_shape
        self.output_shape = (self.units,)
        self.shape = (self.input_shape[0], self.output_shape[0])

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        self._initialize_variables()
        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

    def call(self, circuit):
        circuit.subcircuit(DenseResistorMatrix(self.name, self.in_subcircuit_net_names, self.out_subcircuit_net_names, np.round(1/self.w, 4)))
        circuit.X(self.name, self.name, *self.in_net_names, *self.out_net_names)

    def backward(self, circuit):
        pass

class DiodeLayer(TwoPorts):
    def __init__(self, name, direction, bias_voltage, trainable=False, use_mos=False, model={}):
        self.direction = direction
        self.bias_voltage = bias_voltage
        self.use_mos = use_mos
        self.model = model
        super().__init__(None, name, layer_kind='diode_layer', trainable=trainable)

    def _set_layer_shapes(self):
        self.units = self.previous_layer.output_shape[0]
        self.input_shape = self.previous_layer.output_shape
        self.output_shape = self.input_shape
        self.shape = (self.input_shape[0], self.output_shape[0])

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()

        self.voltage_drops = np.zeros(shape=self.shape) # not sure about what this is here for

        if self.previous_layer.layer_kind != 'diode_layer':
            super()._generate_subcircuit_net_names()
            self._generate_net_names()
        else:
            self.in_subcircuit_net_names = self.previous_layer.in_subcircuit_net_names
            self.out_subcircuit_net_names = self.previous_layer.out_subcircuit_net_names
            self.in_net_names = self.previous_layer.in_net_names
            self.out_net_names = self.previous_layer.out_net_names

        if self.trainable:
            self._generate_subcircuit_current_net_names()

    def _generate_net_names(self):
        self.in_net_names = self.previous_layer.out_net_names
        self.out_net_names = self.previous_layer.out_net_names

    def _generate_subcircuit_current_net_names(self):
        self.current_net_names = [f"v.x{self.name}.v{i+1}" for i in range(self.units)]

    def call(self, circuit):
        if self.use_mos:
            circuit.subcircuit(MOSDiode(self.name, self.in_subcircuit_net_names, self.out_subcircuit_net_names, self.direction, self.bias_voltage, self.model))
        else:
            circuit.subcircuit(Diode(self.name, self.in_subcircuit_net_names, self.out_subcircuit_net_names, self.direction, self.bias_voltage, self.model))
        circuit.X(self.name, self.name, *self.in_net_names)

class AmplificationLayer(TwoPorts):
    def __init__(self, name, gain):
        self.gain = gain
        super().__init__(None, name, layer_kind='amplification_layer')

    def _set_layer_shapes(self):
        self.units = self.previous_layer.output_shape[0]
        self.input_shape = self.previous_layer.output_shape
        self.output_shape = self.input_shape
        self.shape = (self.input_shape[0], self.output_shape[0])

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

    def call(self, circuit):
        circuit.subcircuit(BidirectionalAmplifier(self.name, self.in_subcircuit_net_names, self.out_subcircuit_net_names, self.gain))
        circuit.X(self.name, self.name, *self.in_net_names, *self.out_net_names)

class CurrentLayer(OnePort):
    def __init__(self, name):
        super().__init__(None, name, layer_kind='current_layer')

    def _set_layer_shapes(self):
        self.units = self.previous_layer.output_shape[0]
        self.output_shape = self.previous_layer.output_shape
        self.shape = (self.output_shape[0], )

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        super()._generate_subcircuit_net_names()
        self._generate_net_names()

    def _generate_net_names(self):
        self.out_net_names = self.previous_layer.out_net_names

    def call(self, circuit, I):
        circuit.subcircuit(CurrentSources(self.name, self.out_subcircuit_net_names, I))
        circuit.X(self.name, self.name, *self.out_net_names)

class ConcatenateLayer(BaseLayer):
    def __init__(self, name):
        super().__init__(None, name, layer_kind='concatenate_layer')

    def _generate_net_names(self, inputs):
        self.in_net_names = [net_name for layer in inputs for net_name in layer.out_net_names]
        self.out_net_names = self.in_net_names
        self.units = len(self.out_net_names)

    def _set_layer_shapes(self, inputs):
        input_shape = np.array(inputs[0].output_shape)
        for previous_layer in inputs[1:]:
            input_shape += np.array(previous_layer.output_shape)
        input_shape = tuple(input_shape)

        self.input_shape = input_shape
        self.output_shape = input_shape
        self.shape = (self.input_shape, self.output_shape)

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes(inputs)
        self._generate_net_names(inputs)

    def call(self, circuit):
        pass

## class LocallyConnected2D:
    #Removed for now due to some bugs
