import numpy as np
from ..one_terminal_devices import OneTerminal
from ...simulation import GROUND
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

    def _define_external_nets(self):
        self.out_net_names = self.parent.out_net_names

    def stamp(self, block, inputs, sample_index):
        # zero current in the free phase; the nudge sets it with alter
        self.sources = block.array(
            [block.current_source(i + 1, net, GROUND, -0.0) for i, net in enumerate(np.ravel(self.out_net_names))],
            self.shape,
        )

    def get_layer_spice_output(self, sim_obj):
        return sim_obj.voltages(self.out_net_names)

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

    def nudge(self, I):
        """The sources to change and their values for injecting currents `I`."""
        I = np.asarray(I, dtype=float)
        if self.fold:
            # each output is a pair of nodes: push one down and the other up
            return self.sources, np.stack([-I, I], axis=1).ravel()
        return self.sources, -I
