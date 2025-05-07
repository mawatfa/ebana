import numpy as np
from ..two_terminal_devices import TwoTerminals
from ...netlist_generators.weight_layers import DenseResistorMatrix, DenseResistorMatrixMeasuringCurrent
from ...utils.generating_functions import generate_names_from_shape
from ...schedules import LearningRateSchedule, ConstantLearningRateSchedule


class DenseLayer(TwoTerminals):
    def __init__(
        self,
        name:str,
        units:int,
        initializer=None,
        lr:float|LearningRateSchedule=0.0,
        trainable:bool=True,
        save_sim_data:bool=True,
        grad_func=None,
        weight_upata_func=None,
        save_currents=False,
    ):
        self.initializer = initializer
        self.lr = lr
        self.save_currents = save_currents
        super().__init__(
            name,
            units,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_upata_func,
        )

        if isinstance(lr, float):
            self.lr = ConstantLearningRateSchedule(lr)
        else:
            self.lr = lr

        if not self.save_currents:
            self.netlist_generator = DenseResistorMatrix
        else:
            self.netlist_generator = DenseResistorMatrixMeasuringCurrent

    def _define_shapes(self) -> None:
        self.input_shape = self.parent.output_shape # type: ignore
        self.output_shape = (self.units,)
        self.shape = (self.input_shape[0], self.output_shape[0]) # type: ignore

    def _define_internal_branches(self):
        spice_letter = "v"
        prefix = f"{spice_letter}.x{self.name}.{spice_letter}"
        self.row_branch_names = generate_names_from_shape(self.input_shape, prefix + "r", use_underscore=False)
        self.col_branch_names = generate_names_from_shape(self.output_shape, prefix + "c", use_underscore=False)


    def _build(self):
        self.w = self.initializer.initialize_weights(shape=self.shape)

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):

        circuit.subcircuit(
            self.netlist_generator(
                self.name,
                self.in_subcircuit_net_names,
                self.out_subcircuit_net_names,
                np.round(1 / self.w, 4),
            )
        )
        circuit.X(self.name, self.name, *self.in_net_names, *self.out_net_names)

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}

        spec = {
            "voltages": {
                "input": self.input_shape,
                "output": self.output_shape,
                }
            }

        if self.save_currents:
            spec["currents"] =  {
                "r": self.input_shape,
                "c": self.output_shape,
            }

        return spec

    def get_batch_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        return {
            "w": self.shape
        }

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:

            input_node_voltages = sim_obj.read_simulation_voltages(self.in_net_names)
            input_node_voltages_reshaped = input_node_voltages.reshape(len(self.in_net_names), 1)
            output_node_voltages = sim_obj.read_simulation_voltages(self.out_net_names)

            self.voltages[phase] = {
                "input": input_node_voltages,
                "output": output_node_voltages,
                "drop": input_node_voltages_reshaped - output_node_voltages,
            }

            if self.save_currents:
                self.currents[phase] = {
                    "r": sim_obj.read_simulation_currents(self.row_branch_names),
                    "c": sim_obj.read_simulation_currents(self.col_branch_names)
                }


    def get_phase_data(self):
        if not self.save_currents:
            return {"voltages": self.voltages}
        else:
            return {"voltages": self.voltages, "currents": self.currents}


    def get_batch_data(self):
        return {"w": self.w}

    def get_variables(self) -> dict:
        return {"w": self.w}

    def set_variables(self, optimizer_state):
        self.w = optimizer_state["w"]

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        return self.voltages[phase]["drop"]

    def default_grad_func(self, free, nudge):
        return (nudge) ** 2 - (free) ** 2

    def default_weight_update_func(self, gradient, epoch_num, batch_num, num_batches):
        step = epoch_num * num_batches + batch_num

        # Compute new weight
        w = self.w -  self.lr(step) * gradient

        # Calculate the maximum allowed change (50% of w)
        max_change =  0.5 * abs(self.w)

        # Limit the update to no more than 10% of the original value
        delta = w - self.w
        delta = delta.clip(-max_change, max_change)
        self.w = self.initializer.clip_conductances(self.w + delta)


class DenseLayerBehavioral:
    pass


class DenseLayerMemristor:
    pass


class DenseLayerMemristorConductance:
    pass
