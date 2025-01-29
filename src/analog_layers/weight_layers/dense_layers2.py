import numpy as np
from ..two_terminal_devices import TwoTerminals
from ...netlist_generators.weight_layers import DenseResistorMatrix, DenseResistorMatrixMeasuringCurrent
from ...utils.generating_functions import generate_names_from_shape


class DenseLayer(TwoTerminals):
    def __init__( self, name:str, units:int, trainable:bool=False, save_sim_data:bool=True, grad_func=None, weight_upata_func=None):
        super().__init__(
            name,
            units,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_upata_func,
        )

    def _define_shapes(self) -> None:
        """
        Define the input and output shape of the layer.
        """

    def _define_internal_nets(self) -> None:
        """
        Net names for use in the creation of the SPICE subcircuit.
        """

    def _define_external_nets(self) -> None:
        """
        Net names for connecting this layer with the other layers.
        """

    def _define_internal_branches(self) -> None:
        """
        Branch names for measuring currents, if needed.
        """

    def _build(self) -> None:
        """
        Initialization of variables for training, if needed.
        """

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index) -> None:
        """
        Generate a subcircuit instance.
        """

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        """
        Returns the dimensions of the data to be saved at the end of a phase.
        """

    def get_batch_data_spec(self) -> dict:
        """
        Returns the dimensions of the data to be saved at the end of a batch.
        """

    def store_phase_data(self, sim_obj, phase):
        """
        Stores the simulation result for a particular phase.
        """


    def get_phase_data(self):
        """
        Returns the phase variable where the phase data is stored.
        """

    def get_batch_data(self):
        """
        Returns the dimensions of the data to be saved at the end of a batch.
        """

    def get_variables(self) -> dict:
        """
        Returns the state dictionary to be saved in the optimizer.
        """

    def set_variables(self, optimizer_state):
        """
        Restore state of internal variables from the saved state.
        """

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        """
        Specifies the data to be returned for use in the calculation of the gradient.
        """

    def default_grad_func(self, free, nudge):
        """
        Calculates and returns the gradient using the phase data.
        """

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        """
        Specifies how the internal weight variable is modified using the gradient.
        """


