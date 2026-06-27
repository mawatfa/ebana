import numpy as np
from ..two_terminal_devices import TwoTerminals
from ...schedules import LearningRateSchedule, as_schedule
from ...constraints import MaxRelativeChange, as_constraint


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
        weight_update_func=None,
        save_currents=False,
        constraint=MaxRelativeChange(0.5),
    ):
        self.initializer = initializer
        self.constraint = as_constraint(constraint)
        self.lr = lr
        self.save_currents = save_currents
        super().__init__(
            name,
            units,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_update_func,
        )

        self.lr = as_schedule(lr)

    def _define_shapes(self) -> None:
        self.input_shape = self.parent.output_shape # type: ignore
        self.output_shape = (self.units,)
        self.shape = (self.input_shape[0], self.output_shape[0]) # type: ignore

    def _build(self):
        if self.initializer is None:
            raise ValueError(f"layer '{self.name}' needs an initializer")
        self.w = self.initializer(self.shape)

    def stamp(self, block, inputs, sample_index):
        R = np.round(1 / self.w, 4)
        rows, cols = list(self.in_net_names), list(self.out_net_names)
        if self.save_currents:
            # zero-volt sources at both ends of each line measure its current
            row_nodes = [block.node(f"row_{j + 1}") for j in range(len(rows))]
            col_nodes = [block.node(f"col_{i + 1}") for i in range(len(cols))]
            self.row_sources = block.array(
                [block.voltage_source(f"r{j + 1}", net, node, 0) for j, (net, node) in enumerate(zip(rows, row_nodes))],
                self.input_shape,
            )
            self.col_sources = block.array(
                [block.voltage_source(f"c{i + 1}", node, net, 0) for i, (net, node) in enumerate(zip(cols, col_nodes))],
                self.output_shape,
            )
            rows, cols = row_nodes, col_nodes
        for i in range(len(cols)):
            for j in range(len(rows)):
                block.resistor(f"{j + 1}_{i + 1}", rows[j], cols[i], R[j][i])

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

            input_node_voltages = sim_obj.voltages(self.in_net_names)
            input_node_voltages_reshaped = input_node_voltages.reshape(len(self.in_net_names), 1)
            output_node_voltages = sim_obj.voltages(self.out_net_names)

            self.voltages[phase] = {
                "input": input_node_voltages,
                "output": output_node_voltages,
                "drop": input_node_voltages_reshaped - output_node_voltages,
            }

            if self.save_currents:
                self.currents[phase] = {
                    "r": sim_obj.currents(self.row_sources),
                    "c": sim_obj.currents(self.col_sources)
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

    def default_weight_update_func(self, update, step):
        self.w = self.constraint(self.w - self.lr(step) * update, self.w)
