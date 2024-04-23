#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
import src.analoglayers as Layers
import PySpice
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
        if model.simulator == 'xyce':
            PySpice.Spice.Simulation.CircuitSimulator.DEFAULT_SIMULATOR = "xyce-serial"


    def build_circuit(self, X, I, idx):
        """
        Build the circuit netlist

        X:   dictionary of inputs
        I:   current to be injected
        idx: index of input sample
        """
        self.circuit = Circuit("new circuit")

        for layer in self.model.computation_graph:
            if isinstance(layer, Layers.InputVoltageLayer):
                layer.call(self.circuit, X[layer.name][idx])
            elif isinstance(layer, Layers.BiasVoltageLayer):
                layer.call(self.circuit, layer.bias_voltage)
            elif isinstance(layer, Layers.CurrentLayer):
                layer.call(self.circuit, I[layer.name])
            else:
                layer.call(self.circuit)
        #with open('readme.txt', 'w') as f:
        #    f.write(str(self.circuit))
        #print(self.circuit)


    def simulate_circuit(self):
        """
        Run an operating-point analysis of the circuit using ngspice
        """
        # run operating point analysis
        simulator = self.circuit.simulator(**SIMULATOR_PARAMS)
        self.analysis = simulator.operating_point()

        if self.model.simulator == "ngspice":
            # destroy the circuit after simulation, otherwise we'll run out of memory
            ngspice = simulator.factory(self.circuit).ngspice
            ngspice.remove_circuit()
            ngspice.destroy()


    def read_simulation_voltages(self, net_names):
        """
        Read node voltages after simulation

        net_names: names of the nodes
        """
        voltages = np.zeros(len(net_names))
        for i, net_name in enumerate(net_names):
            voltages[i] = float(self.analysis.nodes[net_name])

        # with open("voltages.txt", "w") as file1:
        #     file1.writelines(str(self.analysis.nodes))

        # with open("currents.txt", "w") as file1:
        #     file1.writelines(str(self.analysis.branches))
        return voltages

    def read_simulation_currents(self, branch_names):
        """
        Read branch currents after simulation

        branch_names: names of the branches
        """
        currents = np.zeros(len(branch_names))
        for i, branch_name in enumerate(branch_names):
            currents[i] = float(self.analysis.branches[branch_name])
        return currents


    def get_voltage_drops(self, phase):
        """
        Get the voltage drop across each conductance

        @param phase: training phase
        """
        for layer in self.model.computation_graph:
            # TODO: rewrite this part so that it works irrespective of the layer type
            if layer.trainable and layer.layer_type in ['dense_layer']:

                if layer.layer_type == 'dense_layer':
                    input_nodes = self.read_simulation_voltages(layer.in_net_names)
                    input_nodes = input_nodes.reshape(len(layer.in_net_names), 1)
                    output_nodes = self.read_simulation_voltages(layer.out_net_names)
                    # TODO: do I need to do this here?
                    if layer.save_output_voltage:
                        layer.output_voltages = output_nodes

                voltage_drop = input_nodes - output_nodes

                if phase == 'free':
                    layer.voltage_drops -=  layer.update_equation(voltage_drop)
                else:
                    layer.voltage_drops +=  layer.update_equation(voltage_drop)

    def get_source_currents(self, phase):
        for layer in self.model.computation_graph:
            if layer.trainable and layer.layer_type in ["input_voltage_layer", "bias_voltage_layer", "diode_layer"]:
                    for i, net_name in enumerate(layer.current_net_names):
                        if phase == 'free':
                            layer.voltage_drops[i] += float(self.analysis.branches[net_name])
                        else:
                            layer.voltage_drops[i] -= float(self.analysis.branches[net_name])

    def get_power_parameters(self, voltage_node_names, current_branch_names):
        result = np.zeros(shape=(2, len(voltage_node_names)))
        result[0] = self.read_simulation_voltages(voltage_node_names)
        result[1] = self.read_simulation_currents(current_branch_names)
        return result
