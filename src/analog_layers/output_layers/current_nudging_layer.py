import numpy as np
from ..one_terminal_devices import OneTerminal
from ...netlist_generators import DCCurrentSources
from ...activations import set_layer_activation

class CurrentNudgingLayer(OneTerminal):
    def __init__(self, name, fold=False, activation=""):
        """
        This layer defines currents sources that inject current into the network
        """
        super().__init__(name=name)
        self.fold = fold
        self.activation = self.set_activation(activation)

    def set_activation(self, activation):
        return set_layer_activation(activation)

    def _define_shapes(self):
        self.units = self.parent.output_shape[0]
        self.output_shape = self.parent.output_shape
        self.shape = (self.output_shape[0],)

    def _define_internal_nets(self):
        self.out_subcircuit_net_names = self.parent.out_subcircuit_net_names

    def _define_external_nets(self):
        self.out_net_names = self.parent.out_net_names

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        current = np.zeros(shape=self.shape)
        circuit.subcircuit(DCCurrentSources(self.name, self.out_subcircuit_net_names, -current))
        circuit.X(self.name, self.name, *self.out_net_names)

    def get_layer_spice_output(self, sim_obj):
        return sim_obj.read_simulation_voltages(self.out_net_names)

    def calculate_layer_output(self, output_node_voltages):
        """
        Calculate the prediction as the difference between the two columns
        of the output_node_voltages array.
        """
        if self.fold:
            output_node_voltages = output_node_voltages.reshape(-1, 2)
            return output_node_voltages[:, 0] - output_node_voltages[:, 1]
        else:
            return output_node_voltages

    def calculate_layer_prediction(self, prediction):
        return self.activation(prediction)

    def call_nudging(self, circuit, I):
        alter_commands = []
        if self.fold:
            for idx in range(len(I)):
                alter_commands.append(f"alter I.X{self.name}.I{2*idx+1}={-I[idx]}")
                alter_commands.append(f"alter I.X{self.name}.I{2*idx+2}={I[idx]}")
        else:
            for idx in range(len(I)):
                alter_commands.append(f"alter I.X{self.name}.I{idx+1}={-I[idx]}")

        return alter_commands
