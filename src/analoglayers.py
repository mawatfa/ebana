#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
from .netlistgenerator import VoltageSources, CurrentSources
from .netlistgenerator import DenseResistorMatrix
from .netlistgenerator import BidirectionalAmplifier
from .netlistgenerator import DiodeBehavioral, DiodeReal, DiodeMOS

#######################################################################
#                              NN Layers                              #
#######################################################################


class BaseLayer:
    def __init__(self, units, name, layer_type, trainable=False):
        """
        This is the base layer that contains basic attributes that all classes must have

        units       : the number of components
        name        : the (unique) name of the layer
        layer_type  : a string to identify the layer
        trainable   : specifies whether the layer parameters should be trainable
        """
        self.units = units
        self.name = name.lower()
        self.layer_type = layer_type
        self.trainable = trainable

        self.save_output_voltage = False
        self.save_power_params = False
        self.built = False

        self.input_shape = None
        self.output_shape = None
        self.shape = None

        # every layer should know its parents and its children
        self.parent = []
        self.children = []

    def define_connections(self, inputs):
        """
        This function updates the parents to know of their children
        """
        # remember your parents
        self.parent = inputs

        # let your parents know you're their child
        if isinstance(inputs, list):
            for parent in inputs:
                parent.children.append(self)
        else:
            self.parent.children.append(self)

    def __call__(self, inputs, mode=None):
        """
        This is the method that gets called when the layers are being defined
        """

        # build the layer if it is not already built
        if not self.built:
            self.build(inputs)
            self.built = True

        # return reference to layer that we can pass to other layers
        return self


class OneTerminal(BaseLayer):
    def __init__(self, units, name, layer_type="one_terminal", trainable=False):
        """
        This layer represents two terminal components, where one of the terminals is connected to ground
        """
        super().__init__(units, name, layer_type, trainable)

    def _generate_subcircuit_net_names(self):
        """
        This function generates the net names used to connect components within the subcircuit
        """
        self.out_subcircuit_net_names = [f"P{i+1}" for i in range(self.units)]

    def _generate_net_names(self):
        """
        This function generates the net names used to connect a layer to another layer
        """
        self.out_net_names = [f"n_{self.name}_{i+1}" for i in range(self.units)]

    def _generate_subcircuit_current_net_names(self, spice_letter):
        """
        This function generates the names of the branches in order to retrieve the current from the simulation output
        """
        self.current_net_names = [
            f"{spice_letter}.x{self.name}.{spice_letter}{i+1}"
            for i in range(self.units)
        ]


class TwoTerminals(BaseLayer):
    def __init__(self, units, name, layer_type="two_terminals", trainable=False):
        """
        This layer represents two terminal components
        """
        super().__init__(units, name, layer_type, trainable)

    def _generate_subcircuit_net_names(self):
        """
        This function generates the net names used to connect components within the subcircuit
        """
        self.in_subcircuit_net_names = [f"IN{i+1}" for i in range(self.input_shape[0])]
        self.out_subcircuit_net_names = [f"OUT{i+1}" for i in range(self.output_shape[0])]

    def _generate_net_names(self):
        """
        This function generates the net names used to connect a layer to another layer
        """
        self.in_net_names = self.parent.out_net_names
        self.out_net_names = [f"n_{self.name}_{i+1}" for i in range(self.output_shape[0])]

    def _generate_subcircuit_current_net_names(self, spice_letter):
        """
        This function generates the names of the branches in order to retrieve the current from the simulation output
        """
        self.current_net_names = [
            f"{spice_letter}.x{self.name}.{spice_letter}{i+1}"
            for i in range(self.units)
        ]


class VoltageLayer(OneTerminal):
    def __init__(self, units, name, layer_type="voltage_layer", trainable=False):
        """
        This layer represents a general voltage layer
        """
        super().__init__(units, name, layer_type, trainable)

    def call(self, circuit, V):
        """
        This is the function that gets called when building the netlist
        """
        circuit.subcircuit(VoltageSources(self.name, self.out_subcircuit_net_names, V))
        circuit.X(self.name, self.name, *self.out_net_names)


class InputVoltageLayer(VoltageLayer):
    def __init__(self, units, name, trainable=False, save_power_params=False):
        """
        This layer represents the input to the network in the form of voltage sources

        units              : the number of voltage sources
        name               : the (unique) name of the layer
        trainable          : specifies whether the voltages should be modified during training
        save_power_params  : specifies whether the voltages and currents should be saved during training
        """
        super().__init__(
            units, name, layer_type="input_voltage_layer", trainable=trainable
        )
        self.save_power_params = save_power_params
        self.build()

    def build(self):
        self.built = True
        self.output_shape = (self.units,)
        self.shape = self.output_shape

        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

        if self.save_power_params:
            super()._generate_subcircuit_current_net_names(spice_letter="v")


class BiasVoltageLayer(VoltageLayer):
    def __init__(
        self, units, name, bias_voltage, trainable=False, save_power_params=False
    ):
        """
        This layer represents the the bias voltage added to every node in a layer

        :units            : the number of voltage sources
        :name             : the (unique) name of the layer
        :trainable        : specifies whether the voltages should be modified during training
        :save_power_params: specifies whether the voltages and currents should be saved during training
        """
        super().__init__(
            units, name, layer_type="bias_voltage_layer", trainable=trainable
        )
        self.bias_voltage = bias_voltage
        self.save_power_params = save_power_params
        self.build()

    def build(self):
        self.built = True
        self.output_shape = (self.units,)
        self.shape = self.output_shape

        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

        if self.save_power_params:
            super()._generate_subcircuit_current_net_names(spice_letter="v")


class DenseLayer(TwoTerminals):
    def __init__(self, units, name, initializer, lr=0.001, trainable=True, save_output_voltage=False):
        """
        This layer represents a dense layer, implemented internally as a 2D array of resistors

        :units               : the number of output units
        :name                : the (unique) name of the layer
        :initializer         : an initializer object that specifies how the resistors are initialized
        :lr                  : the learning rate of the layer
        :trainable           : specifies whether the resistances should be modified during training
        :save_output_voltage : specifies whether the voltages of the nodes that are connected to the children layers are to be saved
        """
        super().__init__(units, name, layer_type="dense_layer", trainable=trainable)
        self.initializer = initializer
        self.lr = lr
        self.save_output_voltage = save_output_voltage

    def _set_layer_shapes(self):
        self.input_shape = self.parent.output_shape
        self.output_shape = (self.units,)
        self.shape = (self.input_shape[0], self.output_shape[0])

    def _initialize_variables(self):
        self.w = self.initializer.initialize_weights(shape=self.shape)

    def update_equation(self, voltage_drop):
        return voltage_drop ** 2

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        self._initialize_variables()
        super()._generate_subcircuit_net_names()
        super()._generate_net_names()

    def call(self, circuit):
        circuit.subcircuit(
            DenseResistorMatrix(
                self.name,
                self.in_subcircuit_net_names,
                self.out_subcircuit_net_names,
                np.round(1 / self.w, 4),
            )
        )
        circuit.X(self.name, self.name, *self.in_net_names, *self.out_net_names)

class DiodeLayer(TwoTerminals):
    def __init__(self, name, direction, bias_voltage, kind, param={}, trainable=False, save_power_params=False):
        """
        This layer represents a diode layer, which introduces nonlinearity into the network

        :name                : the (unique) name of the layer
        :direction           : the direction of the diode (up / down)
        :bias_voltage        : a voltage source connected in series with the diode to shift threshold at which the diode turns on
        :kind                : the kind of diode (behavioral, real, mosf)
        :param               : a dictionary of the parameters that describe the behavior of the diode
        :trainable           : specifies whether the bias voltages should be modified during training
        :save_power_params: specifies whether the voltages and currents should be saved during training
        """
        super().__init__(None, name, layer_type="diode_layer", trainable=trainable)
        self.direction = direction
        self.bias_voltage = bias_voltage
        self.kind = kind
        self.param = param
        self.save_power_params = save_power_params

    def _set_layer_shapes(self):
        self.units = self.parent.output_shape[0]
        self.input_shape = self.parent.output_shape
        self.output_shape = self.input_shape
        self.shape = self.output_shape

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()

        if self.parent.layer_type != "diode_layer":
            super()._generate_subcircuit_net_names()
            self._generate_net_names()
        else:
            self.in_subcircuit_net_names = self.parent.in_subcircuit_net_names
            self.out_subcircuit_net_names = self.parent.out_subcircuit_net_names
            self.in_net_names = self.parent.in_net_names
            self.out_net_names = self.parent.out_net_names

        # TODO: see how to remove this from here
        if self.save_power_params:
            self._generate_subcircuit_current_net_names(spice_letter="v")

    def _generate_net_names(self):
        self.in_net_names = self.parent.out_net_names
        self.out_net_names = self.parent.out_net_names

    def call(self, circuit):
        args = [
            self.name,
            self.in_subcircuit_net_names,
            self.out_subcircuit_net_names,
            self.direction,
            self.bias_voltage,
            self.param,
        ]
        if self.kind == "behavioral":
            circuit.subcircuit(DiodeBehavioral(*args))
        elif self.kind == "real":
            circuit.subcircuit(DiodeReal(*args))
        elif self.kind == "mos":
            circuit.subcircuit(DiodeMOS(*args))
        circuit.X(self.name, self.name, *self.in_net_names)


class AmplificationLayer(TwoTerminals):
    def __init__(self, name, param, save_power_params=False):
        """
        This layer represents amplification layer, which boosts the voltage in one direction by `gain` and the current in the other direction by 1/`gain`

        :name                : the (unique) name of the layer
        :param               : a dictionary of the parameters that describe the behavior of the amplifier
        :save_power_params:  : specifies whether the voltages and currents should be saved during training
        """
        super().__init__(None, name, layer_type="amplification_layer")
        self.param = param
        self.save_power_params = save_power_params

    def _set_layer_shapes(self):
        self.units = self.parent.output_shape[0]
        self.input_shape = self.parent.output_shape
        self.output_shape = self.input_shape
        self.shape = (self.input_shape[0], self.output_shape[0])

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        super()._generate_subcircuit_net_names()
        super()._generate_net_names()
        self._generate_subcircuit_current_net_names(spice_letter="b")

    def call(self, circuit):
        circuit.subcircuit(
            BidirectionalAmplifier(
                self.name,
                self.in_subcircuit_net_names,
                self.out_subcircuit_net_names,
                self.param,
            )
        )
        circuit.X(self.name, self.name, *self.in_net_names, *self.out_net_names)


class CurrentLayer(OneTerminal):
    def __init__(self, name):
        """
        This layer defines currents sources that inject current into the network
        """
        super().__init__(None, name, layer_type="current_layer")

    def _set_layer_shapes(self):
        self.units = self.parent.output_shape[0]
        self.output_shape = self.parent.output_shape
        self.shape = (self.output_shape[0],)

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes()
        super()._generate_subcircuit_net_names()
        self._generate_net_names()

    def _generate_net_names(self):
        self.out_net_names = self.parent.out_net_names

    def call(self, circuit, I):
        circuit.subcircuit(CurrentSources(self.name, self.out_subcircuit_net_names, I))
        circuit.X(self.name, self.name, *self.out_net_names)


class ConcatenateLayer(BaseLayer):
    """
    This layer combines the net names of many layers into one
    """

    def __init__(self, name):
        super().__init__(None, name, layer_type="concatenate_layer")

    def _generate_net_names(self, inputs):
        self.in_net_names = [
            net_name for layer in inputs for net_name in layer.out_net_names
        ]
        self.out_net_names = self.in_net_names
        self.units = len(self.out_net_names)

    def _set_layer_shapes(self, inputs):
        input_shape = np.array(inputs[0].output_shape)
        for parent in inputs[1:]:
            input_shape += np.array(parent.output_shape)
        input_shape = tuple(input_shape)

        self.input_shape = input_shape
        self.output_shape = input_shape
        self.shape = (self.input_shape, self.output_shape)

    def build(self, inputs):
        super().define_connections(inputs)
        self._set_layer_shapes(inputs)
        self._generate_net_names(inputs)

    def call(self, circuit):
        return


# class LocallyConnected2D:
# Removed for now due to some bugs
