import numpy as np
from pycircuitsim.simulator import NgSpiceSession, VerilogaModel, simulations

# one ngspice per process
_session = None
_loaded_veriloga = set()


class Simulator:
    """Loads netlists into this process's ngspice and reads results by handle."""

    @property
    def session(self):
        global _session
        if _session is None:
            _session = NgSpiceSession()
        return _session

    def load(self, netlist):
        for path in sorted(netlist.veriloga_files - _loaded_veriloga):
            model = VerilogaModel(module_name="", path=path)
            model.compile_model()
            self.session.send_command(f"osdi {model}")
            _loaded_veriloga.add(path)
        # free the previous circuit, otherwise memory grows
        self.session.reset()
        self.session.load_netlist(netlist.lines())

    def alter(self, elements, values):
        elements, values = np.broadcast_arrays(np.asarray(elements, dtype=object), np.asarray(values, dtype=float))
        for element, value in zip(elements.ravel(), values.ravel()):
            self.session.send_command(f"alter {element.name}={float(value)!r}")

    def solve(self):
        self.session.run(simulations.OP())

    def voltages(self, nets):
        return self._read(nets, str)

    def currents(self, elements):
        def vector(element):
            if not element.branch:
                raise ValueError(f"ngspice has no branch current for '{element.name}'")
            return f"{element.name}#branch"
        return self._read(elements, vector)

    def _read(self, items, vector_name):
        items = np.asarray(items, dtype=object)
        values = np.empty(items.shape)
        for idx, item in np.ndenumerate(items):
            name = vector_name(item)
            data = self.session.get_data(name)
            if data is None:
                raise KeyError(f"ngspice vector '{name}' not found")
            values[idx] = data[0]
        return values
