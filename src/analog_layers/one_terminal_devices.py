import numpy as np
from .base import BaseLayer
from ..utils.generating_functions import generate_names_from_shape

class OneTerminal(BaseLayer):
    """
    Layer for single-terminal devices (one net to ground).
    """

    def _define_shapes(self) -> None:
        if isinstance(self.parent, list):
            raise ValueError("OneTerminal layer supports only one parent.")

        if self.parent is None:
            self.input_shape = None
            self.output_shape = (self.units,)
            self.shape = self.output_shape
        else:
            # The parent's output shapes is our input and output shape
            self.input_shape = self.parent.output_shape  # type: ignore
            self.output_shape = self.parent.output_shape # type: ignore
            self.shape = self.output_shape               # type: ignore
            self.units = (np.prod(self.input_shape),)

    def _define_internal_nets(self) -> None:
        self.out_subcircuit_net_names = generate_names_from_shape(self.shape, "P")

    def _define_external_nets(self) -> None:
        prefix = f"n_{self.name}"
        self.out_net_names = generate_names_from_shape(self.shape, prefix)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index: int) -> None:
        """
        Implementation for building a one-terminal device subcircuit.
        """
        pass
