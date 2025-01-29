from PySpice.Spice.Netlist import SubCircuit

class DCCurrentSources(SubCircuit):
    def __init__(self, name: str, net_names: list, c_arr):
        SubCircuit.__init__(self, name, *net_names)
        for i in range(len(c_arr)):
            self.I(i + 1, net_names[i], self.gnd, c_arr[i])
