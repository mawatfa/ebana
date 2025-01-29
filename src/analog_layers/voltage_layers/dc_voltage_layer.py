import numpy as np
from ..one_terminal_devices import OneTerminal
from ...netlist_generators import DCVoltageSources
from ...utils.generating_functions import generate_names_from_shape

class DCVoltageLayer(OneTerminal):
    """
    Provides DC voltages to the circuit (input or bias).
    """
    def __init__(self, name, units, input_voltages=None, trainable=False, save_sim_data=False):
        self.input_voltages = input_voltages
        super().__init__(name, units, trainable=trainable, save_sim_data=save_sim_data)
        self.build()

    def _define_internal_branches(self):
        spice_letter = "v"
        prefix = f"{spice_letter}.x{self.name}.{spice_letter}"
        self.branch_names = generate_names_from_shape(self.shape, prefix)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        voltages = self.input_voltages if self.input_voltages is not None else spice_input_dict[self.name][sample_index]
        circuit.subcircuit(DCVoltageSources(self.name, self.out_subcircuit_net_names, voltages))
        circuit.X(self.name, self.name, *self.out_net_names)

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "drop": self.output_shape,
        }

    def get_batch_data_spec(self) -> dict:
        return {}

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
            self.currents[phase] = {
                "drop": sim_obj.read_simulation_currents(self.branch_names)
            }

    def get_phase_data(self):
        return self.currents
