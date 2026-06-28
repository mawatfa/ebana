import numpy as np

from ..activations import set_layer_activation
from ..simulation import GROUND
from .base import Layer


class CurrentNudging(Layer):
    """
    Reads the input nets, and nudges them with current sources. With
    fold=True every output is the difference of a pair of nets.
    """

    def __init__(self, name, fold=False, activation="", save_sim_data=False):
        super().__init__(name, save_sim_data=save_sim_data)
        self.fold = fold
        self.activation = set_layer_activation(activation)

    def build(self, inputs):
        (x,) = inputs
        if x.ndim != 1:
            raise ValueError(f"CurrentNudging '{self.name}' needs a 1-D input, got shape {x.shape}")
        if self.fold and len(x) % 2:
            raise ValueError(f"CurrentNudging '{self.name}' with fold=True needs an even number of nets, got {len(x)}")
        return x

    def stamp(self, block, data, sample_index):
        # zero current in the free phase; the nudge sets it with alter
        self.sources = block.array(
            [block.current_source(i + 1, net, GROUND, -0.0) for i, net in enumerate(self.output)],
            self.output_shape,
        )

    def get_layer_spice_output(self, sim):
        return sim.voltages(self.output)

    def calculate_layer_output(self, output_node_voltages):
        """With fold=True, each output is the difference of its pair of nets."""
        if self.fold:
            pairs = output_node_voltages.reshape(-1, 2)
            return pairs[:, 0] - pairs[:, 1]
        return output_node_voltages

    def calculate_layer_prediction(self, output):
        return self.activation(output)

    def nudge(self, I):
        """The sources to change and their values for injecting currents `I`."""
        I = np.asarray(I, dtype=float)
        if self.fold:
            # each output is a pair of nodes: push one down and the other up
            return self.sources, np.stack([-I, I], axis=1).ravel()
        return self.sources, -I
