import numpy as np
from ..one_terminal_devices import OneTerminal

# TODO: Needs Readjusting
class SimplexVoltageLayer(OneTerminal):
    def __init__(self, name, vdd, vss, lr, initializer, trainable=False):
        """
        This layer defines currents sources that inject current into the network
        """
        self.lr = lr
        self.vdd = vdd
        self.vss = vss
        self.initializer = initializer
        super().__init__(name=name, trainable=trainable)
        print(self.save_sim_data)

    def _define_shapes(self):
        self.units = self.parent.output_shape[0]
        self.output_shape = self.parent.output_shape
        self.shape = (self.output_shape[0]+1,)

    def _define_internal_nets(self):
        self.out_subcircuit_net_names = self.parent.out_subcircuit_net_names

    def _define_external_nets(self):
        self.out_net_names = self.parent.out_net_names

    def _build(self):
        self.w = self.initializer.initialize_weights(shape=(1,self.output_shape[0]+1))[0]

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index):
        current = np.zeros(shape=(self.units, ))
        circuit.subcircuit(SimplexVoltage(self.name, self.out_subcircuit_net_names, self.vdd, self.vss, self.w))
        circuit.X(self.name, self.name, *self.out_net_names)

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
            simulation_voltages = sim_obj.read_simulation_voltages(self.out_net_names)
            input_node_voltages = np.zeros(shape=(len(simulation_voltages)+1, ))
            output_node_voltages = np.zeros(shape=(len(simulation_voltages)+1, ))
            input_node_voltages[0] = self.vdd
            input_node_voltages[1:] = simulation_voltages
            output_node_voltages[:-1] = simulation_voltages

            self.voltages[phase] = {
                "input": input_node_voltages,
                "output": output_node_voltages,
                "drop": input_node_voltages - output_node_voltages,
            }

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        return self.voltages[phase]["drop"]

    def default_grad_func(self, free, nudge):
        return (nudge) ** 2 - (free) ** 2

    def default_weight_update_func(self, gradient, epoch_num, batch_num):
        w = self.w - self.lr * gradient
        self.w = self.initializer.clip_conductances(w)
