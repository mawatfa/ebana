from .base import BaseLayer
from ..utils.generating_functions import generate_names_from_shape

class TwoTerminals(BaseLayer):
    """
    Layer for two-terminal devices.
    """

    def _define_internal_nets(self) -> None:
        # Subcircuit net names for inputs and outputs
        self.in_subcircuit_net_names = generate_names_from_shape(self.input_shape, "IN")
        self.out_subcircuit_net_names = generate_names_from_shape(self.output_shape, "OUT")

    def _define_external_nets(self) -> None:
        # Input net names come from parent
        self.in_net_names = self.parent.out_net_names  # type: ignore
        # Generate output for two terminal devices
        self.out_net_names = generate_names_from_shape(self.output_shape, f"n_{self.name}")

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index: int) -> None:
        """
        Implementation for building a two-terminal device subcircuit.
        """
        # Implementation depends on hardware environment
        pass
