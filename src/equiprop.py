#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
from multiprocessing import Lock
from .spicesimulation import Spice

#######################################################################
#                  equilibirum propagation algorithm                  #
#######################################################################

class EqProp:

    mutex = Lock()

    def __init__(self, model):
        self.model = model

    def free_phase(self, X, idx):
        # no current is injected in the free phase
        I_injected = {}
        for k, v in self.model.output_nodes.items():
            I_injected[k] = np.zeros(shape=(v, 2))

        self.spice.build_circuit(X, I_injected, idx)
        self.spice.simulate_circuit()

        # calculate voltage drops
        self.spice.get_voltage_drops(phase='free')
        self.spice.get_source_currents(phase='free')

        # get output node voltages
        self.output_node_voltages = {}
        for layer in self.model.outputs:
            self.output_node_voltages[layer.name] = self.spice.read_simulation_voltages(layer.out_net_names).reshape(-1, 2)


    def nudging_phase(self, X, idx, I_injected):

        self.spice.build_circuit(X, I_injected, idx)
        self.spice.simulate_circuit()

        ## calculate voltage drops
        self.spice.get_voltage_drops(phase='nudging')
        self.spice.get_source_currents(phase='nudging')


    def train(self, X, Y, batch_size, process_num, N_PROCESSES, layers_dict=None, loss_dict=None, mode="training"):
        """
        This implements a training loop according to the equilibirum propagation algorithm

        @param X:           dictionary of inputs
        @param Y:           list of outputs
        @param beta:        beta
        @param process_num: the process number
        @loss_fn:           loss function for calculating the loss
        @layer_dict:       a dictionary of numpy arrays that are shared by the processes
        """

        # initialize the voltages stored in the layers to 0 before a process starts
        for layer in self.model.computation_graph:
            if layer.trainable:
                layer.voltage_drops = np.zeros(layer.shape)

        # maintain a running sum of the loss of each sample in the dataset
        if mode == "training":
            batch_loss = {}
            for k, v in self.model.output_nodes.items():
                batch_loss[k] = np.zeros(shape=(v, ))
        else:
            batch_output_node_voltages = {}
            #process_batch_size = int(np.floor((batch_size - process_num) / N_PROCESSES))
            for k, v in self.model.output_nodes.items():
                batch_output_node_voltages[k] = np.zeros(shape=(batch_size, v))

        # simulate all samples in the dataset for the given process
        for idx in range(process_num, batch_size, N_PROCESSES):

            # create spice object
            self.spice = Spice(self.model)

            # run free phase
            self.free_phase(X, idx)

            # calculate loss and current that needs to be injected in nudging phase
            if mode == "training":
                I_injected, losses = {}, {}
                for k, v in self.model.output_nodes.items():
                    losses[k], I_injected[k] = self.model.loss_fn(self.output_node_voltages[k], Y[k][idx], self.model.beta)

                # add sample loss to batch loss
                for k, v in self.model.output_nodes.items():
                    batch_loss[k] += losses[k]

                # run nudging phase
                self.nudging_phase(X, idx, I_injected)

            else:
                for k, v in self.model.output_nodes.items():
                    prediction =  self.model.loss_fn(self.output_node_voltages[k], mode=mode)
                    batch_output_node_voltages[k][idx] = prediction

        # write result to dictionary shared with other processes
        self.mutex.acquire()

        if mode == "training":
            # for the layer parameters
            for layer in self.model.computation_graph:
                if layer.trainable:
                    layers_dict[layer.name] += layer.voltage_drops

            # for the losses
            for k, v in self.model.output_nodes.items():
                loss_dict[k] += batch_loss[k]

        else:
            for k, v in self.model.output_nodes.items():
                loss_dict[k] += batch_output_node_voltages[k]

        self.mutex.release()


    def train_inputs(self, X, Y, batch_size, process_num, N_PROCESSES, layers_dict=None, loss_dict=None, dataset_dict=None, mode="training"):
        """
        This implements a training loop according to the equilibirum propagation algorithm

        @param X:           dictionary of inputs
        @param Y:           list of outputs
        @param beta:        beta
        @param process_num: the process number
        @loss_fn:           loss function for calculating the loss
        @layer_dict:       a dictionary of numpy arrays that are shared by the processes
        """

        # initialize the voltages stored in the layers to 0 before a process starts
        for layer in self.model.computation_graph:
            if layer.trainable:
                layer.voltage_drops = np.zeros(layer.shape)

        # maintain a running sum of the loss of each sample in the dataset
        if mode == "training":
            batch_loss = {}
            for k, v in self.model.output_nodes.items():
                batch_loss[k] = np.zeros(shape=(v, ))
        else:
            batch_output_node_voltages = {}
            #process_batch_size = int(np.floor((batch_size - process_num) / N_PROCESSES))
            for k, v in self.model.output_nodes.items():
                batch_output_node_voltages[k] = np.zeros(shape=(batch_size, v))

        dataset = {}
        for k, v in X.items():
            dataset[k] = np.zeros(shape=(self.model.dataset_size, v.shape[-1]))

        # simulate all samples in the dataset for the given process
        for idx in range(process_num, batch_size, N_PROCESSES):

            # create spice object
            self.spice = Spice(self.model)

            # run free phase
            self.free_phase(X, idx)

            # calculate loss and current that needs to be injected in nudging phase
            if mode == "training":
                I_injected, losses = {}, {}
                for k, v in self.model.output_nodes.items():
                    losses[k], I_injected[k] = self.model.loss_fn(self.output_node_voltages[k], Y[k][idx], self.model.beta)

                # add sample loss to batch loss
                for k, v in self.model.output_nodes.items():
                    batch_loss[k] += losses[k]

                # run nudging phase
                self.nudging_phase(X, idx, I_injected)

                for layer in self.model.computation_graph:
                    if layer.layer_kind in ['input_voltage_layer']:
                        dataset[layer.name][idx] = layer.voltage_drops
                        layer.voltage_drops = np.zeros(layer.shape)

            else:
                for k, v in self.model.output_nodes.items():
                    prediction = self.output_node_voltages[k][:,1] - self.output_node_voltages[k][:,0]
                    batch_output_node_voltages[k][idx] = prediction

        # write result to dictionary shared with other processes
        self.mutex.acquire()

        if mode == "training":
            # for the layer parameters
            for layer in self.model.computation_graph:
                if layer.trainable:
                    layers_dict[layer.name] += layer.voltage_drops

            # for the losses
            for k, v in self.model.output_nodes.items():
                loss_dict[k] += batch_loss[k]


            for k, v in dataset.items():
                dataset_dict[k] += v

        else:
            for k, v in self.model.output_nodes.items():
                loss_dict[k] += batch_output_node_voltages[k]

        self.mutex.release()


    def evaluate(self, X, Y, batch_size, process_num, N_PROCESSES, layers_dict):
        self.train(X, Y, batch_size, process_num, N_PROCESSES, mode="evaluating", loss_dict=layers_dict)
