import numpy as np
from ..two_terminal_devices import TwoTerminals
from ...simulation import GROUND
from ...schedules import as_schedule


class AmplificationLayer(TwoTerminals):
    def __init__(self, name, param, lr=0.0, trainable=False, save_sim_data=False):
        self.param = param
        self.lr = as_schedule(lr)
        super().__init__(name, trainable=trainable, save_sim_data=save_sim_data)

    def _define_shapes(self):
        self.input_shape = self.parent.output_shape  # type: ignore
        self.output_shape = self.input_shape
        self.shape = self.input_shape

    def _build(self) -> None:
        self.w = np.zeros(self.shape) + self.param["gain"]

    def stamp(self, block, inputs, sample_index):
        shift, gain = self.param["shift"], self.param["gain"]
        outputs = []
        for i, (inn, out) in enumerate(zip(np.ravel(self.in_net_names), np.ravel(self.out_net_names)), start=1):
            # drive the output, and draw the matching current from the input so the stage is bidirectional
            amp = block.behavioral_voltage(i, out, GROUND, f"{{{shift} + {gain} * (V({inn}) - {shift})}}")
            block.behavioral_current(f"i{i}", inn, GROUND, f"{{{-1/gain} * I({amp.name})}}")
            outputs.append(amp)
        self.amplifiers = block.array(outputs, self.shape)

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
                "input": sim_obj.voltages(self.in_net_names),
            }
            self.currents[phase] = {
                "drop": sim_obj.currents(self.amplifiers)
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

    def default_weight_update_func(self, update, step):
        # all units share one gain
        w = np.mean(self.w - self.lr(step) * update)
        self.w = np.zeros_like(self.w) + w
        self.param["gain"] = self.w[0]
