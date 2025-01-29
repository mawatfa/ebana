# imports <<<
import numpy as np

import src.analog_layers as Layers
import PySpice
from PySpice.Spice.Netlist import Circuit
# >>>

# simulation parameters <<<
SIMULATOR_PARAMS = {
        "temperature": 27,
        "nominal_temperature": 27
        }
# >>>

class Spice:
    def __init__(self, model):
        self.model = model

    # build netlist <<<
    def build_circuit_free(self, dataset_input_dict, idx=0):
        self.circuit = Circuit("ebana_circuit")
        for layer in self.model.computation_graph:
            layer.build_spice_subcircuit(self.circuit, dataset_input_dict, idx)

    def build_circuit_nudge(self, nudge_current):
        self.circuit_update = []
        for layer in self.model.computation_graph:
            if isinstance(layer, Layers.CurrentNudgingLayer):
                self.circuit_update.extend(layer.call_nudging(self.circuit, nudge_current[layer.name]))

    def write_netlist(self, filename, dataset_input_dict, idx=0):
        self.build_circuit_free(dataset_input_dict, idx)
        if filename:
            simulator = self.circuit.simulator(**SIMULATOR_PARAMS)
            with open(filename, "w") as f:
                f.write(str(simulator))
    # >>>

    # simulate circuit <<<
    def simulate_circuit(self, mode="", circuit_update=None, batch_size=None):
        """
        Run an operating-point analysis of the circuit
        """
        # run operating point analysis
        simulator = self.circuit.simulator(**SIMULATOR_PARAMS)

        if self.model.simulation_kind == "op":
            self.analysis = simulator.operating_point(circuit_update=circuit_update)

        elif self.model.simulation_kind == "tran":

            delay = self.model.sampling_time["delay"]
            duration = self.model.sampling_time["duration"]
            transient = self.model.sampling_time["transition"]

            if not batch_size:
                batch_size = self.model.batch_size

            step_time = 1e-6
            end_time = delay + batch_size*duration + (batch_size - 1)*transient
            use_initial_condition=True

            self.analysis = simulator.transient(step_time=step_time, end_time=end_time, use_initial_condition=use_initial_condition, circuit_update=circuit_update)

        if self.model.simulator == "ngspice" and (circuit_update or mode in ["evaluating", "predicting"]):
            # destroy the circuit after simulation, otherwise we'll run out of memory
            ngspice = simulator.factory(self.circuit).ngspice
            ngspice.remove_circuit()
            ngspice.destroy()
    # >>>

    # read voltages <<<
    def read_simulation_voltages(self, net_names):
        if self.model.simulation_kind == "tran":
            return self.read_transient_simulation_voltages(net_names)
        else:
            return self.read_dc_simulation_voltages(net_names)

    def read_dc_simulation_voltages(self, net_names):
        voltages = np.zeros(len(net_names))
        for i, net_name in enumerate(net_names):
            voltages[i] = float(self.analysis.nodes[net_name])
        return voltages

    def read_transient_simulation_voltages(self, net_names, batch_size=None):
        if not batch_size:
            time_values = sample_times(**self.model.sampling_time, n_cycles=self.model.batch_size)
        else:
            time_values = sample_times(**self.model.sampling_time, n_cycles=batch_size)

        # get the indices of the time values according to the sample times
        idx_array = np.zeros_like(time_values, dtype=int)
        for i, t in enumerate(time_values):
            differences = np.abs(np.array(self.analysis.time) - t[0])
            index1 = np.argmin(differences)
            differences = np.abs(np.array(self.analysis.time) - t[1])
            index2 = np.argmin(differences)
            idx_array[i] = [index1, index2]

        free_voltages = np.zeros(shape=(len(net_names), len(time_values)))
        nudge_voltages = np.zeros(shape=(len(net_names), len(time_values)))

        for i, net_name in enumerate(net_names):
            for j in range(len(idx_array)):
                begin, end = idx_array[j]
                mid = begin + (end - begin) // 2
                free_voltages[i][j] = np.mean(np.array(self.analysis.nodes[net_name])[begin:mid])
                nudge_voltages[i][j] = np.mean(np.array(self.analysis.nodes[net_name])[mid+1:end])

        return free_voltages, nudge_voltages

    # >>>

    # read currents <<<
    def read_simulation_currents(self, branch_names):
        if self.model.simulation_kind == "tran":
            raise NotImplementedError("Not Implemented Yet !")
        else:
            return self.read_dc_simulation_currents(branch_names)

    def read_dc_simulation_currents(self, branch_names):
        currents = np.zeros(len(branch_names))
        for i, branch_name in enumerate(branch_names):
            currents[i] = float(self.analysis.branches[branch_name])
        return currents
    # >>>

    # get simulation data <<<
    def get_sim_data(self, phase):
        if self.model.simulation_kind == "tran":
            pass

        else:
            for layer in self.model.computation_graph:
                if layer.save_sim_data:
                    layer.store_phase_data(self, phase)
    # >>>
