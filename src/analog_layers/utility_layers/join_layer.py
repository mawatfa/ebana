import numpy as np
from ..base import BaseLayer

class JoinNetNamesLayer(BaseLayer):
    """
    Joins the net names of all parent layers to match the first parent's net names.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def _define_internal_nets(self):
        pass

    def _define_shapes(self):
        if len(self.parent) < 2:
            raise ValueError("JoinNetNamesLayer requires at least two layers.")
        self.input_shape = self.parent[0].output_shape
        self.output_shape = self.parent[0].output_shape

    def _define_external_nets(self):
        reference_out_net_names = self.parent[0].out_net_names
        for p in self.parent[1:]:
            p.out_net_names = reference_out_net_names
        self.in_net_names = self.parent[0].in_net_names
        self.out_net_names = reference_out_net_names

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        pass
