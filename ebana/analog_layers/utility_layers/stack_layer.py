import numpy as np
from ..base import BaseLayer


class StackLayer(BaseLayer):
    """
    Stacks outputs from multiple parent layers along a new axis.
    """

    def __init__(self, name: str, axis: int = 0):
        """
        Initialize the StackLayer.

        Parameters:
            name: Name of the layer.
            axis: Axis along which to stack the outputs (default is 0).
        """
        super().__init__(name=name)
        self.axis = axis

    def _define_internal_nets(self):
        pass

    def _define_shapes(self):
        """
        Set the input_shape and output_shape by stacking parent output shapes along the specified axis.
        """
        # Calculate new output shape
        stacked_dim = len(self.parent)

        if stacked_dim < 2:
            raise ValueError("StackLayer requires at least two parents.")

        # Ensure all parents have the same shape
        parent_shapes = [p.output_shape for p in self.parent]

        # Check for shape consistency across all parents
        reference_shape = parent_shapes[0]
        for shape in parent_shapes:
            if shape != reference_shape:
                raise ValueError("All parent layers must have the same output shape to stack.")

        # Set input_shape to match the reference shape (single parent shape)
        self.input_shape = tuple(reference_shape)

        # Compute output shape by inserting the new axis at the correct position
        output_shape = list(reference_shape)
        output_shape.insert(self.axis if self.axis >= 0 else len(reference_shape) + 1, stacked_dim)
        self.output_shape = tuple(output_shape)

    def _define_external_nets(self):
        """
        Define the input and output net names by stacking parent net names.
        """
        self.in_net_names = []
        for p in self.parent:
            self.in_net_names.extend(p.out_net_names)

        # Use numpy to stack net names
        self.out_net_names = np.stack([np.array(p.out_net_names) for p in self.parent], axis=self.axis)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        """
        No SPICE subcircuit is needed here, since we only pass net names forward.
        """
        pass
