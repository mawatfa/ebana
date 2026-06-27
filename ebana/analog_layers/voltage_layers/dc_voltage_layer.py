import numpy as np
from ..one_terminal_devices import OneTerminal
from ...simulation import GROUND

class DCVoltageLayer(OneTerminal):
    """
    Provides DC voltages to the circuit (input or bias).
    """
    def __init__(self, name, units, input_voltages=None, trainable=False, save_sim_data=False):
        self.input_voltages = input_voltages
        super().__init__(name, units, trainable=trainable, save_sim_data=save_sim_data)
        self.build()

    def stamp(self, block, inputs, sample_index):
        voltages = self.input_voltages if self.input_voltages is not None else inputs[self.name][sample_index]
        nets, voltages = np.ravel(self.out_net_names), np.ravel(voltages)
        self.sources = block.array(
            [block.voltage_source(i + 1, net, GROUND, v) for i, (net, v) in enumerate(zip(nets, voltages))],
            self.shape,
        )

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
                "drop": sim_obj.currents(self.sources)
            }

    def get_phase_data(self):
        return self.currents
