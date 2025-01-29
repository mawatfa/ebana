import numpy as np
from ..base import BaseLayer

class ReshapeLayer(BaseLayer):
    def __init__(self, name:str, shape:tuple):
        """
        Args:
            name: Name of the layer
            shape: Desired output shape. The product of these dimensions must match the number of net names from the parent.
        """
        super().__init__(name=name)
        self.shape = shape

    def _define_shapes(self):
        # The parent's output_shape is our input shape
        self.input_shape = self.parent.output_shape  # type: ignore

        # 2. For the specific case of shape=(-1,)
        if len(self.shape) == 1 and self.shape[-1] == -1:
            self.output_shape = (np.prod(self.input_shape),)
        # 2. the shape the user request
        else:
            self.output_shape = self.shape

    def _define_internal_nets(self):
        # This layer does not need extra internal nets
        pass

    def _define_external_nets(self):
        # 1) in_net_names come from the parent's out_net_names (list of strings)
        self.in_net_names = self.parent.out_net_names  # type: ignore

        # 2) Reshape in_net_names into the new shape
        self.out_net_names = self.in_net_names.reshape(self.shape)
        # exit()


    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        """
        No SPICE subcircuit is needed here, since we only pass net names forward.
        """
        pass
