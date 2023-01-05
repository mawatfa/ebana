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


# class DenseResistorMatrix(SubCircuit):
#     def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, W):
#         SubCircuit.__init__(self, name, *in_subcircuit_net_names, *out_subcircuit_net_names)
#         for i in range(len(out_subcircuit_net_names)):
#             for j in range(len(in_subcircuit_net_names)):
#                 inn = in_subcircuit_net_names[j]
#                 out = out_subcircuit_net_names[i]
#                 self.B(f"{str(j+1)}_{str(i+1)}", inn, out, current_expression=f"{{{1 / W[j][i]} * sinh( V({inn},{out}) / 0.25 )}}")


class LocallyConnected2DResistorMatrix(SubCircuit):
    def __init__(self, name, unique_input_names, in_subcircuit_net_names, out_subcircuit_net_names, W):
        SubCircuit.__init__(self, name, *unique_input_names, *out_subcircuit_net_names)
        for i, out_name in enumerate(out_subcircuit_net_names):
            for j, in_name in enumerate(in_subcircuit_net_names[i]):
                self.R(f"{str(j+1)}{str(i+1)}", in_name, out_name, W[j][i])

#-----------------------------------------------------------------------------#
#                                 Diodes                                      #
#-----------------------------------------------------------------------------#

class DiodeBehavioral(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, direction, bias, param):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        if direction == 'down':
            for i in range(len(out_subcircuit_net_names)):
                out = in_subcircuit_net_names[i]
                VTH = param["VTH"]
                RON = param["RON"]
                self.B(f"d{i+1}", out, self.gnd, current_expression=f"{{max(0, (V({out}) - {VTH + bias[i]}) / {RON} )}}")
        elif direction == 'up':
            for i in range(len(out_subcircuit_net_names)):
                out = in_subcircuit_net_names[i]
                VTH = param["VTH"]
                RON = param["RON"]
                self.B(f"u{i+1}", self.gnd, out, current_expression=f"{{max(0, -(V({out}) + {VTH + bias[i]}) / {RON} )}}")

class DiodeReal(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, direction, bias, param):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        model_name = param['model_name']
        self.raw_spice = ".include " + param['path']
        if direction == 'down':
            for i in range(len(out_subcircuit_net_names)):
                self.D(i+1, in_subcircuit_net_names[i], f"N{i+1}", model=model_name)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])
        else:
            for i in range(len(out_subcircuit_net_names)):
                self.D(i+1, f"N{i+1}", in_subcircuit_net_names[i], model=model_name)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])


class DiodeMOS(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, direction, bias, param):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        model_name = param['model_name']
        length = param['length']
        width = param['width']
        self.raw_spice = ".include " + param['path']
        if direction == 'down':
            for i in range(len(out_subcircuit_net_names)):
                name = in_subcircuit_net_names[i]
                self.M(i+1, f"N{i+1}", name, name, name, model=model_name, l=length, w=width)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])
        else:
            for i in range(len(out_subcircuit_net_names)):
                self.M(i+1, in_subcircuit_net_names[i], f"N{i+1}", f"N{i+1}", f"N{i+1}", model=model_name, l=length, w=width)
                self.V(i+1, f"N{i+1}", self.gnd, bias[i])

#-----------------------------------------------------------------------------#
#                          Bidirectional Amplifier                            #
#-----------------------------------------------------------------------------#
class BidirectionalAmplifier(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, out_subcircuit_net_names, param):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names, *out_subcircuit_net_names)
        shift = param["shift"]
        gain = param["gain"]
        for i in range(len(out_subcircuit_net_names)):
            inn = in_subcircuit_net_names[i]
            out = out_subcircuit_net_names[i]
            #self.VCVS(i+1, out_subcircuit_net_names[i], self.gnd, in_subcircuit_net_names[i], self.gnd, gain)
            #self.B(i+1, in_subcircuit_net_names[i], self.gnd, current_expression=f"{-1/gain} * I(E{i+1})")

            ## this works fine
            self.B(i+1, out, self.gnd, voltage_expression=f"{{{shift} + {gain} * (V({inn}) - {shift})}}")
            self.B(f"i{i+1}", inn, self.gnd, current_expression=f"{{{-1/gain} * I(B{i+1})}}")
            ##

            #########################################################################
            # implementing the bidrectional amplifier with more relastic components #
            #########################################################################
#             self.raw_spice = """
# .include /home/medwatt/.ltspice/TransistorModels/NMOS_VTL.inc
# .include /home/medwatt/.ltspice/TransistorModels/PMOS_VTL.inc
#              """

#             ######################
#             # inverter amplifier #
#             ######################
#             inverter_gain = 8
#             nmos_model = "NMOS_VTL"
#             pmos_model = "PMOS_VTL"

#             # sources
#             VDD = 1
#             self.V(f"dd{i+1}", f"vdd_plus{i+1}", self.gnd, VDD)

#             # inverter
#             self.M(f"p{i+1}", f"inv_out{i+1}", inn, f"vdd_plus{i+1}", f"vdd_plus{i+1}", model=pmos_model, l="50n", w="160n")
#             self.M(f"n{i+1}", f"inv_out{i+1}", inn, self.gnd, self.gnd, model=nmos_model, l="50n", w="90n")

#             #####################
#             # current amplifier #
#             #####################

#             opamp_gain = 1000
#             RS = 1000

#             self.V(f"sen{i+1}", f"inv_out{i+1}", out, 0)

#             # self.R(f"NS{i+1}", f"NS{i+1}", self.gnd, RS)
#             # self.M(f"NS{i+1}", inn, f"NG{i+1}", f"NS{i+1}", self.gnd, model=nmos_model, l="100n", w="500n")
#             # self.B(f"NS{i+1}", f"NG{i+1}", self.gnd, voltage_expression=f"max(0, min(1, {opamp_gain} * ( V(N_op_p{i+1}) - V(NS{i+1}) )))")
#             # self.H(f"NS{i+1}", f"N_op_p{i+1}", self.gnd, f"Vsen{i+1}", RS / inverter_gain)

#             self.R(f"PS{i+1}", f"vdd_plus{i+1}", f"PS{i+1}", RS)
#             self.M(f"PS{i+1}", inn, f"PG{i+1}", f"PS{i+1}", f"vdd_plus{i+1}", model=pmos_model, l="100n", w="500n")
#             self.B(f"PS{i+1}", f"PG{i+1}", self.gnd, voltage_expression=f"max(0, min(1, {opamp_gain} * ( V(P_op_p2_{i+1}) - V(PS{i+1}) )))")
#             self.V(f"PS1_{i+1}", f"P_op_p2_{i+1}", f"P_op_p1_{i+1}", VDD)
#             self.H(f"PS{i+1}", f"P_op_p1_{i+1}", self.gnd, f"Vsen{i+1}", RS / inverter_gain)

class CurrentSources(SubCircuit):
    def __init__(self, name, in_subcircuit_net_names, I):
        SubCircuit.__init__(self, name, *in_subcircuit_net_names)
        for i in range(I.shape[0]):
            self.I(2*i+1, self.gnd, in_subcircuit_net_names[2*i], I[i][0])
            self.I(2*i+2, self.gnd, in_subcircuit_net_names[2*i+1], I[i][1])
