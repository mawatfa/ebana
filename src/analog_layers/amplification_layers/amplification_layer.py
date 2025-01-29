import numpy as np
from ..two_terminal_devices import TwoTerminals
from ...netlist_generators import BidirectionalAmplifier
from ...utils.generating_functions import generate_names_from_shape


class AmplificationLayer(TwoTerminals):
    def __init__(self, name, param, lr=0.0, trainable=False):
        self.param = param
        self.lr = lr
        super().__init__(name, trainable=trainable)

    def _define_shapes(self):
        self.input_shape = self.parent.output_shape  # type: ignore
        self.output_shape = self.input_shape
        self.shape = self.input_shape

    def _define_internal_branches(self):
        spice_letter = "b"
        prefix = f"{spice_letter}.x{self.name}.{spice_letter}"
        self.branch_names = generate_names_from_shape(self.shape, prefix, use_underscore=False)

    def _build(self) -> None:
        self.w = np.zeros(self.shape) + self.param["gain"]

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        circuit.subcircuit(
            BidirectionalAmplifier(
                self.name,
                self.in_subcircuit_net_names.flatten(),
                self.out_subcircuit_net_names.flatten(),
                self.param,
            )
        )
        circuit.X(
            self.name,
            self.name,
            *self.in_net_names.flatten(),
            *self.out_net_names.flatten(),
        )

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "voltage": {
                "input": self.input_shape,
            },
            "current": {
                "drop": self.output_shape
                }
            }

    def get_phase_data(self):
        return {
                "voltage": self.voltages,
                "current": self.currents
        }

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
            self.voltages[phase] = {
                "input": sim_obj.read_simulation_voltages(self.in_net_names),
            }
            self.currents[phase] = {
                "drop": sim_obj.read_simulation_currents(self.branch_names)
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
        # self.w = optimizer_state["w"]
        pass


    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        return - self.currents[phase]["drop"] * self.voltages[phase]["input"]

    def default_grad_func(self, free, nudge):
        return nudge - free

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        # self.w -= self.lr * gradient

        w = np.mean(self.w - self.lr * gradient)
        self.w = np.zeros_like(self.w) + w

        self.param["gain"] = self.w[0]
        print(self.w)
