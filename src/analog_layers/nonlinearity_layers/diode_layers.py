import numpy as np
from ..one_terminal_devices import OneTerminal
from ...netlist_generators import DiodeBehavioral, DiodeReal
from ...utils.generating_functions import generate_names_from_shape


class DiodeLayer(OneTerminal):
    def __init__(
        self,
        name: str,
        direction: str,
        bias_voltage: float,
        kind: str,
        lr: float = 0.0,
        param: dict = {},
        trainable: bool = False,
        save_sim_data: bool = False,
    ):
        self.direction = direction
        self.bias_voltage = bias_voltage
        self.kind = kind
        self.lr = lr
        self.param = param
        super().__init__(name, trainable=trainable, save_sim_data=save_sim_data)

    def _define_external_nets(self):
        super()._define_external_nets()
        self.out_net_names = self.parent.out_net_names

    def _define_internal_branches(self):
        spice_letter = "v"
        prefix = f"{spice_letter}.x{self.name}.{spice_letter}"
        self.branch_names = generate_names_from_shape(self.shape, prefix, use_underscore=False)

    def _build(self):
        self.w = np.zeros(shape=self.shape) + self.bias_voltage

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        args = [
            self.name,
            self.out_subcircuit_net_names.flatten(),
            self.direction,
            self.w.flatten(),
            self.param,
        ]
        if self.kind == "behavioral":
            circuit.subcircuit(DiodeBehavioral(*args))
        elif self.kind == "real":
            circuit.subcircuit(DiodeReal(*args))
            circuit.raw_spice = ".lib " + self.param['path']

        circuit.X(self.name, self.name, *self.out_net_names.flatten())

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "current": {
                "drop": self.output_shape
                }
            }

    def get_phase_data(self):
        return {"current": self.currents}

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
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
        return self.currents[phase]["drop"]

    def default_grad_func(self, free, nudge):

        off_current_scale_factor = 1e5
        current_threshold = 1e-12

        current = self.currents["free"]["drop"]
        cur_mag_mask = np.where(np.abs(current) > current_threshold, 1, 0)

        if self.direction == "up":
            diode_cond_dir_mask = np.where(current < 0, 1, -off_current_scale_factor)
            grad = nudge - free
        else:
            diode_cond_dir_mask = np.where(current > 0, 1, -off_current_scale_factor)
            grad = free - nudge

        return grad * diode_cond_dir_mask * cur_mag_mask

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        self.w -= self.lr * gradient
        # w = np.mean(self.w - self.lr * gradient)
        # self.w = np.zeros_like(self.w) + w
