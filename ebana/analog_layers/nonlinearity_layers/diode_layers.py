import numpy as np
from ..one_terminal_devices import OneTerminal
from ...simulation import GROUND
from ...schedules import as_schedule


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
        self.lr = as_schedule(lr)
        self.param = param
        super().__init__(name, trainable=trainable, save_sim_data=save_sim_data)

    def _define_external_nets(self):
        super()._define_external_nets()
        self.out_net_names = self.parent.out_net_names

    def _build(self):
        self.w = np.zeros(shape=self.shape) + self.bias_voltage

    def stamp(self, block, inputs, sample_index):
        if self.kind == "real":
            block.include(self.param["path"])
        sources = []
        for i, (out, bias) in enumerate(zip(np.ravel(self.out_net_names), np.ravel(self.w)), start=1):
            node = block.node(f"n{i}")
            # the diode conducts from `out` to its bias node ("down") or the other way ("up")
            anode, cathode = (out, node) if self.direction == "down" else (node, out)
            if self.kind == "behavioral":
                VTH, RON, ROFF = self.param["VTH"], self.param["RON"], self.param["ROFF"]
                if self.direction == "down":
                    x = f"(V({out}) - V({node}) - {VTH})"
                else:
                    x = f"-(V({out}) + V({node}) + {VTH})"
                block.behavioral_current(f"d{i}", anode, cathode, f"{{ max(({x}/{ROFF}), ({x}/{RON})) }}")
            else:
                block.diode(i + 1, anode, cathode, model=self.param["model_name"])
            sources.append(block.voltage_source(i, node, GROUND, bias))
        self.bias_sources = block.array(sources, self.shape)

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
                "drop": sim_obj.currents(self.bias_sources)
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

    def default_weight_update_func(self, update, step):
        self.w -= self.lr(step) * update
        # w = np.mean(self.w - self.lr * gradient)
        # self.w = np.zeros_like(self.w) + w
