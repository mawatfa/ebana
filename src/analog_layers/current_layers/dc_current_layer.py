import numpy as np
from ..one_terminal_devices import OneTerminal
from ...netlist_generators import DCCurrentSources


class DCCurrentLayer(OneTerminal):
    """
    Provides DC voltages to the circuit (input or bias).
    """

    def __init__(
        self,
        name,
        units=-1,
        lr=0,
        initialize_currents=True,
        trainable=False,
        save_sim_data=False,
        grad_func=None,
        weight_update_func=None,
    ):
        self.lr = lr
        self.initialize_currents = initialize_currents
        super().__init__(
            name,
            units,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_update_func,
        )

    def _define_external_nets(self):
        super()._define_external_nets()
        self.out_net_names = self.parent.out_net_names

    def _build(self):
        if self.initialize_currents:
            self.w = np.zeros(shape=self.shape)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):

        # if self.name in ["cd1r", "cd1c"]:
        #     self.w = spice_input_dict[self.name][sample_index]

        self.w = (
            self.w
            if self.initialize_currents
            else spice_input_dict[self.name][sample_index]
        )

        circuit.subcircuit(DCCurrentSources(self.name, self.out_subcircuit_net_names, self.w))
        circuit.X(self.name, self.name, *self.out_net_names)

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "voltages": {
                "drop": self.output_shape
                }
            }

    def get_phase_data(self):
        return {"voltages": self.voltages}

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
            self.voltages[phase] = {
                "drop": sim_obj.read_dc_simulation_voltages(self.out_net_names)
            }

    def get_batch_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "w": self.shape
        }

    def get_batch_data(self) -> dict:
        return {"w": self.w}

    def get_variables(self) -> dict:
        return {"w": self.w}

    def set_variables(self, optimizer_state) -> None:
        self.w = optimizer_state["w"]

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        return self.voltages[phase]["drop"]

    def default_grad_func(self, free, nudge):
        return nudge - free

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        self.w -= self.lr * gradient
