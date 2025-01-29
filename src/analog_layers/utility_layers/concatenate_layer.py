import numpy as np
from ..base import BaseLayer

class ConcatenateLayer(BaseLayer):
    """
    Concatenates outputs from multiple parent layers along a specified axis.
    """

    def __init__(self, name: str, axis: int = 0):
        super().__init__(name=name)
        self.axis = axis

    def _define_internal_nets(self):
        pass

    def _define_shapes(self):
        """
        Sets the input_shape and output_shape by concatenating parent output shapes along the given axis.
        """
        if len(self.parent) < 2:
            raise ValueError("ConcatenateLayer requires at least two parents.")

        parent_shapes = [p.output_shape for p in self.parent]

        # Check shape consistency except along the concatenation axis
        ref_shape = list(parent_shapes[0])
        for shape in parent_shapes[1:]:
            shape_list = list(shape)
            for i in range(len(ref_shape)):
                if i != self.axis and shape_list[i] != ref_shape[i]:
                    raise ValueError("All parent shapes must match along non-concatenation axes.")

        # Compute concatenated shape
        concat_dim = sum([s[self.axis] for s in parent_shapes])
        ref_shape[self.axis] = concat_dim
        self.output_shape = tuple(ref_shape)

        # Input shape is arbitrary reference since all are consistent except for the axis
        self.input_shape = parent_shapes[0]

    def _define_external_nets(self):
        """
        Stacks input net names and concatenates them along the specified axis.
        """
        self.in_net_names = []
        for p in self.parent:
            self.in_net_names.extend(p.out_net_names)

        # Concatenate net names
        all_net_names = [np.array(p.out_net_names, dtype=object) for p in self.parent]
        self.out_net_names = np.concatenate(all_net_names, axis=self.axis)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        pass
