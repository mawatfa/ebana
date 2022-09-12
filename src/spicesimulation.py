#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
from PySpice.Spice.Netlist import Circuit

#######################################################################
#                        simulator parameters                         #
#######################################################################

SIMULATOR_PARAMS = {
        'temperature' : 27,
        'nominal_temperature' : 27
        }

#######################################################################
#                           class defintion                           #
#######################################################################

class Spice:

    def __init__(self, model):
        self.model = model


    def build_circuit(self, X, I, idx):
        """
        Build the circuit netlist

        @param X:   dictionary of inputs
        @param I:   current to be injected
        @param idx: index of input sample
        """
        self.circuit = Circuit("new circuit")

        for layer in self.model.computation_graph:
            if layer.layer_kind in ['input_voltage_layer']:
                layer.call(self.circuit, X[layer.name][idx])
            elif layer.layer_kind in ['bias_voltage_layer']:
                layer.call(self.circuit, layer.bias_voltage)
            elif layer.layer_kind == 'current_layer':
                layer.call(self.circuit, I[layer.name])
            else:
                layer.call(self.circuit)
        #print(self.circuit)


    def simulate_circuit(self):
        """
        Run an operating-point analysis of the circuit using ngspice
        """
        # run operating point analysis
        simulator = self.circuit.simulator(**SIMULATOR_PARAMS)
        self.analysis = simulator.operating_point()

        # destroy the circuit after simulation, otherwise we'll run out of memory
        ngspice = simulator.factory(self.circuit).ngspice
        ngspice.remove_circuit()
        ngspice.destroy()


    def read_simulation_voltages(self, net_names):
        """
        Read node voltages after simulation

        @param net_names: names of the nodes
        """
        voltages = np.zeros(len(net_names))
        for i, net_name in enumerate(net_names):
            voltages[i] = float(self.analysis.nodes[net_name])
        return voltages


    def get_voltage_drops(self, phase):
        """
        Get the voltage drop across each conductance

        @param phase: training phase
        """
        for layer in self.model.computation_graph:
            if layer.trainable and layer.layer_kind in ['dense_layer', "LocallyConnected2D"]:

                if layer.layer_kind == 'dense_layer':
                    input_nodes = self.read_simulation_voltages(layer.in_net_names)
                    input_nodes = input_nodes.reshape(len(layer.in_net_names), 1)
                    output_nodes = self.read_simulation_voltages(layer.out_net_names)

                elif layer.layer_kind == "LocallyConnected2D":
                    input_nodes = np.zeros(shape=(layer.shape))
                    for i in range(layer.shape[0]):
                        input_nodes[i] = self.read_simulation_voltages(layer.in_net_names[i])
                    output_nodes = (self.read_simulation_voltages(layer.out_net_names)).reshape(layer.shape[0],1)

                if phase == 'free':
                    layer.voltage_drops -=  (input_nodes - output_nodes) ** 2
                else:
                    layer.voltage_drops +=  (input_nodes - output_nodes) ** 2

    def get_source_currents(self, phase):
        for layer in self.model.computation_graph:
            if layer.trainable and layer.layer_kind in ["diode_layer", "bias_voltage_layer", "input_voltage_layer"]:
                    for i, net_name in enumerate(layer.current_net_names):
                        if phase == 'free':
                            layer.voltage_drops[i] += float(self.analysis.branches[net_name])
                        else:
                            layer.voltage_drops[i] -= float(self.analysis.branches[net_name])
