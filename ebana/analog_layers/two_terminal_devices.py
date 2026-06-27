from .base import BaseLayer
from ..utils.generating_functions import generate_names_from_shape

class TwoTerminals(BaseLayer):
    """
    Layer for two-terminal devices.
    """

    def _define_internal_nets(self) -> None:
        pass

    def _define_external_nets(self) -> None:
        # Input net names come from parent
        self.in_net_names = self.parent.out_net_names  # type: ignore
        # Generate output for two terminal devices
        self.out_net_names = generate_names_from_shape(self.output_shape, f"n_{self.name}")

