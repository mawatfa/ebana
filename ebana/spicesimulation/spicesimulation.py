# imports <<<
import numpy as np

import ebana.analog_layers as Layers
from pycircuitsim.netlisting import Circuit
from pycircuitsim.simulator import NgSpiceSession, simulations
# >>>

# simulation parameters <<<
SIMULATOR_PARAMS = {
        "temperature": 27,
        "nominal_temperature": 27
        }
# >>>

# one ngspice instance per process, reused across samples <<<
_session = None

def get_session():
    global _session
    if _session is None:
        _session = NgSpiceSession()
    return _session
# >>>

class Spice:
    def __init__(self, model):
        self.model = model

    # build netlist <<<
    def build_circuit_free(self, dataset_input_dict, idx=0):
        self.circuit = Circuit("ebana_circuit")
        self.circuit.raw(f".options TEMP={SIMULATOR_PARAMS['temperature']} TNOM={SIMULATOR_PARAMS['nominal_temperature']}")
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
            with open(filename, "w") as f:
                f.write(str(self.circuit))
    # >>>

    # simulate circuit <<<
    def simulate_circuit(self, mode="", circuit_update=None, batch_size=None):
        # without circuit_update the new circuit is loaded, otherwise the
        # alter commands are applied to the loaded one
        session = get_session()

        if circuit_update is None:
            # free the previous circuit and its plots, otherwise we'll run out of memory
            session.reset()
            session.load_netlist(self.circuit.get_netlist())
        else:
            for command in circuit_update:
                session.send_command(command)

        if self.model.simulation_kind == "op":
            session.run(simulations.OP())

        elif self.model.simulation_kind == "tran":

            delay = self.model.sampling_time["delay"]
            duration = self.model.sampling_time["duration"]
            transient = self.model.sampling_time["transition"]

            if not batch_size:
                batch_size = self.model.batch_size

            step_time = 1e-6
            end_time = delay + batch_size*duration + (batch_size - 1)*transient

            session.run(simulations.Tran(tstep=step_time, tstop=end_time, uic=True))

    def _get_vector(self, name):
        data = get_session().get_data(name)
        if data is None:
            raise KeyError(f"ngspice vector '{name}' not found")
        return data
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
            voltages[i] = self._get_vector(net_name)[0]
        return voltages

    def read_transient_simulation_voltages(self, net_names, batch_size=None):
        if not batch_size:
            time_values = sample_times(**self.model.sampling_time, n_cycles=self.model.batch_size)
        else:
            time_values = sample_times(**self.model.sampling_time, n_cycles=batch_size)

        # get the indices of the time values according to the sample times
        time = self._get_vector("time")
        idx_array = np.zeros_like(time_values, dtype=int)
        for i, t in enumerate(time_values):
            differences = np.abs(time - t[0])
            index1 = np.argmin(differences)
            differences = np.abs(time - t[1])
            index2 = np.argmin(differences)
            idx_array[i] = [index1, index2]

        free_voltages = np.zeros(shape=(len(net_names), len(time_values)))
        nudge_voltages = np.zeros(shape=(len(net_names), len(time_values)))

        for i, net_name in enumerate(net_names):
            node_voltages = self._get_vector(net_name)
            for j in range(len(idx_array)):
                begin, end = idx_array[j]
                mid = begin + (end - begin) // 2
                free_voltages[i][j] = np.mean(node_voltages[begin:mid])
                nudge_voltages[i][j] = np.mean(node_voltages[mid+1:end])

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
            currents[i] = self._get_vector(f"{branch_name}#branch")[0]
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
