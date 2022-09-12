#######################################################################
#                               imports                               #
#######################################################################

from PySpice.Spice.Netlist import SubCircuit
from PySpice.Spice.Library import SpiceLibrary

#######################################################################
#                          generate netlists                          #
#######################################################################

class VoltageSources(SubCircuit):
    def __init__(self, name, out_subcircuit_net_names, V):
        SubCircuit.__init__(self, name, *out_subcircuit_net_names)
        for i in range(len(V)):
            self.V(i+1, out_subcircuit_net_names[i], self.gnd, V[i])

class DenseResistorMatrix(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, W):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names, *out_subcircuit_net_names)
        for i in range(len(out_subcircuit_net_names)):
            for j in range(len(in_subcircuit_net_names)):
                self.R(f"{str(j+1)}_{str(i+1)}", in_subcircuit_net_names[j], out_subcircuit_net_names[i], W[j][i])

class LocallyConnected2DResistorMatrix(SubCircuit):
    def __init__(self, name, unique_input_names, in_subcircuit_net_names, out_subcircuit_net_names, W):
        SubCircuit.__init__(self, name, *unique_input_names, *out_subcircuit_net_names)
        for i, out_name in enumerate(out_subcircuit_net_names):
            for j, in_name in enumerate(in_subcircuit_net_names[i]):
                self.R(f"{str(j+1)}{str(i+1)}", in_name, out_name, W[j][i])

class MOSDiode(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, direction, bias, spice_model={}):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        model_name = spice_model['model_name']
        if spice_model:
            model_name = spice_model['model_name']
            self.raw_spice = ".include " + spice_model['path']
            length = spice_model['length']
            width = spice_model['width']
        else:
            # default NMOS model
            self.model(model_name, 'nmos', Kp=190E-6, Vto=0.57, Lambda=0.16, Gamma=0.50, Phi=0.7)
            length = 1e-6
            width = 10e-6

        if direction == 'down':
            for i in range(len(out_subcircuit_net_names)):
                name = in_subcircuit_net_names[i]
                self.M(i+1, f"N{i+1}", name, name, name, model=model_name, l=length, w=width)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])
        else:
            for i in range(len(out_subcircuit_net_names)):
                self.M(i+1, in_subcircuit_net_names[i], f"N{i+1}", f"N{i+1}", f"N{i+1}", model=model_name, l=length, w=width)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])

class Diode(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, direction, bias, spice_model={}):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        model_name = 'diode-model'
        if spice_model:
            model_name = spice_model['model_name']
            self.raw_spice = ".include " + spice_model['path']
        else:
            # default diode model
            self.model(model_name, 'D', Is=1e-6, RS=0, N=2)

        if direction == 'down':
            for i in range(len(out_subcircuit_net_names)):
                self.D(i+1, in_subcircuit_net_names[i], f"N{i+1}", model=model_name)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])
        else:
            for i in range(len(out_subcircuit_net_names)):
                self.D(i+1, f"N{i+1}", in_subcircuit_net_names[i], model=model_name)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])

class BidirectionalAmplifier(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, gain):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names, *out_subcircuit_net_names)
        for i in range(len(out_subcircuit_net_names)):
            self.VCVS(i+1, out_subcircuit_net_names[i], self.gnd, in_subcircuit_net_names[i], self.gnd, gain)
            self.B(i+1, in_subcircuit_net_names[i], self.gnd, current_expression=f"{-1/gain} * I(E{i+1})")

class CurrentSources(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, I):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        for i in range(I.shape[0]):
            self.I(2*i+1, self.gnd, in_subcircuit_net_names[2*i], I[i][0])
            self.I(2*i+2, self.gnd, in_subcircuit_net_names[2*i+1], I[i][1])
