# imports <<<
import numpy as np

import ebana.analog_layers as Layers
from pycircuitsim.netlisting import Circuit
from pycircuitsim.simulator import NgSpiceSession, simulations
# >>>

# simulation parameters <<<
SIMULATOR_PARAMS = {"temperature": 27, "nominal_temperature": 27}
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
        self.circuit.raw(
            f".options TEMP={SIMULATOR_PARAMS['temperature']} TNOM={SIMULATOR_PARAMS['nominal_temperature']}"
        )
        for layer in self.model.computation_graph:
            layer.build_spice_subcircuit(self.circuit, dataset_input_dict, idx)

    def build_circuit_nudge(self, nudge_current):
        self.circuit_update = []
        for layer in self.model.computation_graph:
            if isinstance(layer, Layers.CurrentNudgingLayer):
                self.circuit_update.extend(
                    layer.call_nudging(self.circuit, nudge_current[layer.name])
                )

    def write_netlist(self, filename, dataset_input_dict, idx=0):
        self.build_circuit_free(dataset_input_dict, idx)
        if filename:
            with open(filename, "w") as f:
                f.write(str(self.circuit))

    # >>>

    # simulate circuit <<<
    def simulate_circuit(self, circuit_update=None):
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

        session.run(simulations.OP())

    def _read(self, names, suffix=""):
        names = np.asarray(names)
        values = np.empty(names.shape)
        for idx, name in np.ndenumerate(names):
            data = get_session().get_data(f"{name}{suffix}")
            if data is None:
                raise KeyError(f"ngspice vector '{name}{suffix}' not found")
            values[idx] = data[0]
        return values

    # >>>

    # read results <<<
    def read_simulation_voltages(self, net_names):
        return self._read(net_names)

    def read_simulation_currents(self, branch_names):
        return self._read(branch_names, "#branch")

    def get_sim_data(self, phase):
        for layer in self.model.computation_graph:
            if layer.save_sim_data:
                layer.store_phase_data(self, phase)

    # >>>
