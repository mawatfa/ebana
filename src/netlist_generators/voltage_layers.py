from PySpice.Spice.Netlist import SubCircuit

class DCVoltageSources(SubCircuit):
    def __init__(self, name: str, net_names: list, v_arr):
        SubCircuit.__init__(self, name, *net_names)
        for i in range(len(v_arr)):
            self.V(i + 1, net_names[i], self.gnd, v_arr[i])
