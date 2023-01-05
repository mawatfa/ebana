#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
from .spicesimulation import Spice

#######################################################################
#                  equilibirum propagation algorithm                  #
#######################################################################

class EqProp:
    def __init__(self, model):
        self.model = model

    def free_phase(self, X, idx):
        """
        This implements the free phase of the EqProp algorithm,
        where no current is injected into the output nodes
        """
        injected_current = {}
        for k, v in self.model.output_nodes.items():
            # the value of the output node is represented by the difference of two nodes
            injected_current[k] = np.zeros(shape=(v, 2))

        self.spice.build_circuit(X, injected_current, idx)
        self.spice.simulate_circuit()

        # calculate voltage drops
        self.spice.get_voltage_drops(phase="free")
        self.spice.get_source_currents(phase="free")

        # TODO: this probably needs to be inside an if clause
        # get output node voltages
        self.output_node_voltages = {}
        for layer in self.model.outputs:
            self.output_node_voltages[layer.name] = self.spice.read_simulation_voltages(layer.out_net_names).reshape(-1, 2)

    def nudging_phase(self, X, idx, injected_current):
        """
        This represents the nudging phase of the EqProp algorithm,
        where loss gradient current is injected into the output nodes
        """
        self.spice.build_circuit(X, injected_current, idx)
        self.spice.simulate_circuit()

        # calculate voltage drops
        self.spice.get_voltage_drops(phase="nudging")
        self.spice.get_source_currents(phase="nudging")

    def train(self, X, Y, batch_size, process_num, N_PROCESSES, mode="training"):
        """
        This implements a training loop according to the equilibirum propagation algorithm

        :X            : dictionary of inputs
        :Y            : list of outputs
        :batch_sizr
        :process_num  : the process number
        """

        if mode == "training":
            # initialize the voltages stored in the layers to 0 before a process starts
            for layer in self.model.computation_graph:
                if layer.trainable:
                    layer.voltage_drops = np.zeros(layer.shape)

            # initialize metrics
            self.model.metrics.initialize_metrics_during_training(batch_size)

            # maintain a running sum of the loss of each sample in the dataset
            batch_loss = {}
            for k, v in self.model.output_nodes.items():
                batch_loss[k] = np.zeros(shape=(v,))

        else: # evaluation mode
            batch_output_node_voltages = {}
            batch_loss = {}
            for k, v in self.model.output_nodes.items():
                batch_output_node_voltages[k] = np.zeros(shape=(batch_size, v))
                batch_loss[k] = np.zeros(shape=(v,))

        # each process handles ~ (batch_size / N_PROCESSES) samples
        for idx in range(process_num, batch_size, N_PROCESSES):

            # create spice object
            self.spice = Spice(self.model)

            # run free phase
            self.free_phase(X, idx)

            # save the metrics after the free phase
            self.model.metrics.save_metrics_during_training(idx)

            # calculate loss and current that needs to be injected in nudging phase
            if mode == "training":
                injected_current, losses = {}, {}
                for k, v in self.model.output_nodes.items():
                    losses[k], injected_current[k] = self.model.loss_fn(
                        self.output_node_voltages[k], Y[k][idx], self.model.beta
                    )

                # add sample loss to batch loss
                for k, v in self.model.output_nodes.items():
                    batch_loss[k] += losses[k]

                # run nudging phase
                self.nudging_phase(X, idx, injected_current)

                # save power metrics after nudging phase
                self.model.metrics.save_metrics_during_training(idx, output_voltages=False, begin=2, end=4)

            else:  # evaluating
                losses = {}
                for k, v in self.model.output_nodes.items():
                    losses[k], _ = self.model.loss_fn(
                        self.output_node_voltages[k], Y[k][idx], self.model.beta
                    )

                for k, v in self.model.output_nodes.items():
                    batch_loss[k] += losses[k]
                    prediction = self.model.loss_fn(
                        self.output_node_voltages[k], mode=mode
                    )
                    batch_output_node_voltages[k][idx] = prediction


        # At the end of the batch, return everything
        if mode == "training":
            process_layer_voltage_drops = {}

            # layer parameters
            for layer in self.model.computation_graph:
                if layer.trainable:
                    process_layer_voltage_drops[layer.name] = layer.voltage_drops

            return {
                "process_layer_voltage_drops": process_layer_voltage_drops,
                "process_losses": batch_loss,
                "process_layer_output_voltage": self.model.metrics.process_training_output_voltages,
                "process_power_params": self.model.metrics.process_training_power_params,
            }

        else:  # evaluating
            return {
                "process_predictions": batch_output_node_voltages,
                "process_losses": batch_loss,
                "process_layer_output_voltage": self.model.metrics.process_training_output_voltages,
                "process_power_params": self.model.metrics.process_training_power_params,
            }

    def evaluate(self, X, Y, batch_size, process_num, N_PROCESSES):
        return self.train(X, Y, batch_size, process_num, N_PROCESSES, mode="evaluating")
