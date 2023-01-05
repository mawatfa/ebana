#######################################################################
#                               imports                               #
#######################################################################
import pickle
import numpy as np
from multiprocessing import Pool
from .equiprop import EqProp

np.random.seed(111)

#######################################################################
#                         parallel processing                         #
#######################################################################

N_PROCESSES = 8

#######################################################################
#                           model defintion                           #
#######################################################################


class Model:
    def __init__(self, inputs, outputs):
        """
        The model is a built as a graph, with some nodes designated as inputs, and others designated as outputs
        """
        self.inputs = inputs    # list of inputs to the network
        self.outputs = outputs  # list of outputs of the network
        self._build_computation_graph(inputs)

    def _build_computation_graph(self, inputs):
        """
        Use breadth-first traversal to make sure that parents are processed before their children
        """
        self.computation_graph = []
        queue = []
        visisted = []

        for input_layer in inputs:
            queue.append(input_layer)

        while queue:
            current_layer = queue[0]
            self.computation_graph.append(current_layer)

            for connected_layer in current_layer.children:
                if connected_layer not in visisted and connected_layer not in queue:
                    queue.append(connected_layer)

            visisted.append(current_layer)
            queue.pop(0)

    # TODO: check if I can find a way to avoid doing this
    def _set_fit_method_variables(self, dataloader):
        # get info from dataloader
        self.dataset_size = dataloader.dataset_size
        self.batch_size = dataloader.batch_size
        self.output_nodes = dataloader.output_size
        self.num_batches = dataloader.num_batches

        # variables to save loss and accuracy
        self.total_loss = {}
        self.test_accuracy = {}
        if self.batch_size != self.dataset_size:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(
                    shape=(self.epochs, self.num_batches, node_size)
                )
                self.test_accuracy[node_name] = np.zeros(
                    shape=(self.epochs, self.num_batches)
                )
        else:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(shape=(self.epochs, node_size))
                self.test_accuracy[node_name] = np.zeros(shape=(self.epochs,))

    # TODO: check if I can find a way to avoid doing this
    def _set_evaluate_method_variables(self, dataset):
        self.dataset = dataset
        self.output_nodes = {}
        for k, v in self.dataset["outputs"].items():
            self.output_nodes[k] = v.shape[1]
            self.dataset_size = v.shape[0]


    #######################################################################
    #                      training/validation methods                    #
    #######################################################################

    def fit(self, *, train_dataloader, beta, epochs, optimizer, metrics, loss_fn, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.beta = beta
        self.epochs = epochs + 1
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss_fn = loss_fn
        self._set_fit_method_variables(self.train_dataloader)

        for epoch_num in range(1, self.epochs):

            self.epoch_num = epoch_num

            for batch_num in range(self.num_batches):

                training_batch = next(self.train_dataloader)
                X = training_batch["inputs"]
                Y = training_batch["outputs"]

                #######################################################################
                #                           multiprocessing                           #
                #######################################################################
                # prepare arguments for the processes
                child_processes_args = []
                for k, v in self.output_nodes.items():
                    for process_num in range(N_PROCESSES):
                        child_processes_args.append(
                            (X, Y, self.batch_size, process_num, N_PROCESSES)
                        )

                # run processes
                pool = Pool(processes=N_PROCESSES)
                results = pool.starmap(EqProp(self).train, child_processes_args)

                #######################################################################
                #                    aggregate loss from processes                    #
                #######################################################################
                process_losses = {}

                for process_num in range(N_PROCESSES):

                    # aggregate voltage drop of trainable layers
                    for layer in self.computation_graph:
                        if layer.trainable:
                            if process_num == 0:
                                layer.voltage_drops = np.zeros(shape=layer.shape)
                            layer.voltage_drops += results[process_num]["process_layer_voltage_drops"][layer.name]

                    # aggregate metrics
                    self.metrics.accumulate_metrics_from_processes(results, process_num)

                    # aggregate loss
                    for k, v in self.output_nodes.items():
                        if process_num == 0:
                            process_losses[k] = np.zeros(shape=(v,))
                        process_losses[k] += results[process_num]["process_losses"][k]
                #######################################################################

                # update the total loss
                if self.batch_size != self.dataset_size:
                    self.total_loss[k][epoch_num][batch_num] = process_losses[k]
                else:
                    for k, v in self.output_nodes.items():
                        self.total_loss[k][epoch_num] = process_losses[k]

                # optimize training parameters
                optimizer.step()

                # print training loss per iteration
                if self.metrics.verbose:
                    self.metrics._print_iteration_loss(epoch_num, batch_num)

                # validate on test/validation dataset
                if self.test_dataloader:
                    test_dataset = next(self.test_dataloader)
                    self.metrics._evaluate_validation(test_dataset, epoch_num, batch_num)

    def predict(self, dataset, training=False):
        """
        Evaluate the circuit to check for accuracy

        @param dataset: dictionary of inputs and outputs
        @param loss_fn: loss function object to calculate loss
        @param training: specify whether we're in training mode or not
        """

        X = dataset["inputs"]
        Y = dataset["outputs"]

        # get dataset size of testing dataset
        dataset_size = Y[list(Y.keys())[0]].shape[0]

        # prepare arguments for the processes
        child_processes_args = []
        for k, v in self.output_nodes.items():
            for process_num in range(N_PROCESSES):
                child_processes_args.append(
                    (X, Y, dataset_size, process_num, N_PROCESSES)
                )

        # run processes
        pool = Pool(processes=N_PROCESSES)
        results = pool.starmap(EqProp(self).evaluate, child_processes_args)

        # aggregate results from the processes
        process_losses = {}
        process_predictions = {}
        for process_num in range(N_PROCESSES):
            # loss
            for k, v in self.output_nodes.items():
                if process_num == 0:
                    process_losses[k] = np.zeros(shape=(v,))
                    process_predictions[k] = np.zeros(shape=(dataset_size, v))
                process_predictions[k] += results[process_num]["process_predictions"][k]
                process_losses[k] += results[process_num]["process_losses"][k]

        return process_losses, process_predictions

    def evaluate(self, dataset, loss_fn=None, verbose=True):
        if loss_fn:
            self.loss_fn = loss_fn
            self._set_evaluate_method_variables(dataset)
        loss, prediction = self.predict(dataset, training=True)
        accuracy = self.metrics._measure_accuracy(dataset, prediction)
        self.metrics._print_accuracy(accuracy, verbose=verbose)
        return loss, accuracy


    #######################################################################
    #                             saving methods                          #
    #######################################################################

    def save_history(self, filepath):
        history = {
            "training_loss": self.total_loss,
            "test_accuracy": self.test_accuracy,
        }
        if self.metrics.save_output_voltages:
            history["output_voltages"] = self.metrics.output_voltages

        if self.metrics.save_power_params:
            history["power_params"] = self.metrics.power_params

        with open(filepath, "wb") as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
