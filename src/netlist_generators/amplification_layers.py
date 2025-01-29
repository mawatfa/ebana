from PySpice.Spice.Netlist import SubCircuit

class BidirectionalAmplifier(SubCircuit):
    def __init__(self, name, in_net_names, out_net_names, param):
        SubCircuit.__init__(self, name, *in_net_names, *out_net_names)
        shift = param["shift"]
        gain = param["gain"]
        for i in range(len(out_net_names)):
            inn = in_net_names[i]
            out = out_net_names[i]
            self.B(f"{i+1}", out, self.gnd, voltage_expression=f"{{{shift} + {gain} * (V({inn}) - {shift})}}")
            self.B(f"i{i+1}", inn, self.gnd, current_expression=f"{{{-1/gain} * I(B{i+1})}}")
