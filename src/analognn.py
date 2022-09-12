#######################################################################
#                               imports                               #
#######################################################################
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Process
from .equiprop import EqProp

np.random.seed(111)

#######################################################################
#                         parallel processing                         #
#######################################################################

N_PROCESSES = 4

#######################################################################
#                           model defintion                           #
#######################################################################

class Model:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self._build_computation_graph(inputs)
        self.computation_graph = inputs[1:] + self._bfs(self.computation_graph, inputs[0])


    def _build_computation_graph(self, inputs):
        self.computation_graph = {}
        queue = []
        explored = []

        for input_layer in inputs:
            queue.append(input_layer)

        while len(queue) > 0:
            current_layer = queue[0]
            self.computation_graph[current_layer] = current_layer.next_layer

            for connected_layer in current_layer.next_layer:
                if connected_layer not in explored and connected_layer not in queue:
                    queue.append(connected_layer)

            explored.append(current_layer)
            queue.pop(0)

        return self.computation_graph


    def _bfs(self, graph, start):
        explored = []
        queue = [start]
        visited= [start]

        while queue:
            node = queue.pop(0)
            explored.append(node)
            neighbours = graph[node]

            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

        return explored


    def _set_fit_method_variables(self, dataloader):
        # get info from dataloader
        self.dataset_size = dataloader.dataset_size
        self.batch_size = dataloader.batch_size
        self.output_nodes = dataloader.output_size
        self.num_batches = dataloader.num_batches

        self.total_loss = {}
        self.test_accuracy = {}
        if self.batch_size != self.dataset_size:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(shape=(self.epochs, self.num_batches, node_size))
                self.test_accuracy[node_name] = np.zeros(shape=(self.epochs, self.num_batches))
        else:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(shape=(self.epochs, node_size))
                self.test_accuracy[node_name] = np.zeros(shape=(self.epochs, ))


    def _set_evaluate_method_variables(self, dataset):
        self.dataset = dataset
        self.output_nodes = {}
        for k, v in self.dataset['outputs'].items():
            self.output_nodes[k] = v.shape[1]
            self.dataset_size = v.shape[0]

    #######################################################################
    #                           metrics methods                           #
    #######################################################################

    def _evaluate_validation(self, dataset, epoch_num, batch_num, verbose, validate_every):
        if 'epoch_num' in validate_every.keys():
            if (epoch_num % validate_every['epoch_num'] == 0) and (batch_num == self.num_batches - 1) or (epoch_num == self.epochs - 1):
                accuracy = self.evaluate(dataset, verbose=verbose)
                for k in self.output_nodes.keys():
                    self.test_accuracy[k][epoch_num] = accuracy[k]

        elif 'batch_num' in validate_every.keys():
            if ((batch_num+1) % validate_every['batch_num'] == 0) or ((epoch_num == self.epochs - 1) and (batch_num == self.num_batches)):
                accuracy = self.evaluate(dataset, verbose=verbose)
                for k in self.output_nodes.keys():
                    self.test_accuracy[k][epoch_num][batch_num] = accuracy[k]

    def _measure_accuracy(self, dataset, prediction):
        target = dataset['outputs']
        accuracy = {}
        for k in self.output_nodes.keys():
            c = self.loss_fn.verify_result(target[k], prediction[k])
            accuracy[k] = np.round(np.mean(c), 2)
        return accuracy


    def _print_accuracy(self, accuracy, verbose=False):
        if verbose:
            print("\n-->Accuracy<--")
            for k in self.output_nodes.keys():
                print(f"    {k}: {accuracy[k]*100:.2f}%")
            print("")


    def _print_iteration_loss(self, epoch_num, batch_num):
        """
        Print iteration loss

        @param iter_n: iteration we're currently in
        """
        loss, loss_mean = {}, {}

        if self.dataset_size != self.batch_size:
            print(f"Iteration {epoch_num}/{batch_num}:")
            for k in self.output_nodes.keys():
                loss[k] = self.total_loss[k][epoch_num][batch_num] / self.batch_size
                loss_mean[k] = np.mean(loss[k])
        else:
            print(f"Iteration {epoch_num}:")
            for k in self.output_nodes.keys():
                loss[k] = self.total_loss[k][epoch_num] / self.dataset_size
                loss_mean[k] =np.mean(loss[k])

        for k in self.output_nodes.keys():
            print(f"    {k}: {loss_mean[k]:.7f} -> {loss[k]}")

    #######################################################################
    #                      training/validation methods                    #
    #######################################################################

    def fit(self, dataloader, beta, epochs, loss_fn, optimizer, verbose=True, test_dataloader=None, validate_every=None):
        self.epochs = epochs + 1
        self.dataloader = dataloader
        self.beta = beta
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.verbose = verbose
        self.test_dataloader = test_dataloader
        self.validate_every = validate_every
        self._set_fit_method_variables(dataloader)

        for epoch_num in range(1, self.epochs):

            for batch_num in range(self.num_batches):

                training_batch = next(dataloader)
                X = training_batch['inputs']
                Y = training_batch['outputs']

                # set up the shared dictionary for the processes
                manager = mp.Manager()
                layers_dict = manager.dict()
                loss_dict = manager.dict()

                # created a shared_dict to save layer parameters from all processes
                for layer in self.computation_graph:
                    if layer.trainable:
                        layers_dict[layer.name] = np.zeros(shape=layer.shape)

                # created a shared_dict to save lossesfrom all processes
                for k, v in self.output_nodes.items():
                    loss_dict[k] = np.zeros(shape=(v, ))

                # fork the processes
                child_processes = []
                for process_num in range(N_PROCESSES):
                    p = Process(target=EqProp(self).train, args=[X, Y, self.batch_size, process_num, N_PROCESSES, layers_dict, loss_dict])
                    p.start()
                    child_processes.append(p)

                # wait until all processes are done
                for process in child_processes:
                    process.join()

                # get back the layer voltages from the processes
                for layer in self.computation_graph:
                    if layer.trainable:
                        layer.voltage_drops = layers_dict[layer.name]

                # get back the losses from all processes
                if self.batch_size != self.dataset_size:
                    self.total_loss[k][epoch_num][batch_num] = loss_dict[k]
                else:
                    for k, v in self.output_nodes.items():
                        self.total_loss[k][epoch_num] = loss_dict[k]

                # optimize training parameters
                optimizer.step(X)

                # print training loss per iteration
                if verbose:
                    self._print_iteration_loss(epoch_num, batch_num)

                # validate on test/validation dataset
                if self.test_dataloader:
                    test_dataset = next(self.test_dataloader)
                    self._evaluate_validation(test_dataset, epoch_num, batch_num, verbose, validate_every)

    def fit_inputs(self, dataloader, beta, epochs, loss_fn, optimizer, verbose=True, test_dataloader=None, validate_every=None):
        self.epochs = epochs + 1
        self.dataloader = dataloader
        self.beta = beta
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.verbose = verbose
        self.test_dataloader = test_dataloader
        self.validate_every = validate_every
        self._set_fit_method_variables(dataloader)

        for epoch_num in range(1, self.epochs):

            for batch_num in range(self.num_batches):

                training_batch = next(dataloader)
                X = training_batch['inputs']
                Y = training_batch['outputs']

                # set up the shared dictionary for the processes
                manager = mp.Manager()
                layers_dict = manager.dict()
                loss_dict = manager.dict()
                dataset_dict = manager.dict()

                # created a shared_dict to save layer parameters from all processes
                for layer in self.computation_graph:
                    if layer.trainable:
                        layers_dict[layer.name] = np.zeros(shape=layer.shape)

                # created a shared_dict to save lossesfrom all processes
                for k, v in self.output_nodes.items():
                    loss_dict[k] = np.zeros(shape=(v, ))

                # created a shared_dict to save lossesfrom all processes
                for k, v in self.dataloader.dataset['inputs'].items():
                    dataset_dict[k] = np.zeros(shape=v.shape)

                # fork the processes
                child_processes = []
                for process_num in range(N_PROCESSES):
                    p = Process(target=EqProp(self).train_inputs, args=[X, Y, self.batch_size, process_num, N_PROCESSES, layers_dict, loss_dict, dataset_dict])
                    p.start()
                    child_processes.append(p)

                # wait until all processes are done
                for process in child_processes:
                    process.join()

                # get back the layer voltages from the processes
                for layer in self.computation_graph:
                    if layer.trainable:
                        layer.voltage_drops = layers_dict[layer.name]

                # get back the losses from all processes
                if self.batch_size != self.dataset_size:
                    self.total_loss[k][epoch_num][batch_num] = loss_dict[k]
                else:
                    for k, v in self.output_nodes.items():
                        self.total_loss[k][epoch_num] = loss_dict[k]

                # optimize training parameters
                #optimizer.step(X)

                for layer in self.computation_graph:
                    if layer.layer_kind in ['input_voltage_layer']:
                        self.dataloader.dataset['inputs'][layer.name] -= 50000 * dataset_dict[layer.name]

                # print training loss per iteration
                if verbose:
                    self._print_iteration_loss(epoch_num, batch_num)

                # validate on test/validation dataset
                if self.test_dataloader:
                    test_dataset = next(self.test_dataloader)
                    self._evaluate_validation(test_dataset, epoch_num, batch_num, verbose, validate_every)


    def predict(self, dataset, training=False):
        """
        Evaluate the circuit to check for accuracy

        @param dataset: dictionary of inputs and outputs
        @param loss_fn: loss function object to calculate loss
        @param training: specify whether we're in training mode or not
        """

        X = dataset['inputs']
        Y = dataset['outputs']

        # get dataset size of testing dataset
        dataset_size = Y[list(Y.keys())[0]].shape[0]

        # set up the shared dictionary for the processes
        manager = mp.Manager()
        layers_dict = manager.dict()

        for k, v in self.output_nodes.items():
            layers_dict[k] = np.zeros(shape=(dataset_size, v))

        # fork the processes
        child_processes = []
        for process_num in range(N_PROCESSES):
            p = Process(target=EqProp(self).evaluate, args=[X, Y, dataset_size, process_num, N_PROCESSES, layers_dict])
            p.start()
            child_processes.append(p)

        # wait until all processes are done
        for process in child_processes:
            process.join()

        # get back the output from all all processes
        prediction_dict = {}
        for k, v in self.output_nodes.items():
            prediction_dict[k] = layers_dict[k]

        return prediction_dict


    def evaluate(self, dataset, loss_fn=None, verbose=True):
        if loss_fn:
            self.loss_fn = loss_fn
            self._set_evaluate_method_variables(dataset)
        prediction = self.predict(dataset, training=True)
        accuracy = self._measure_accuracy(dataset, prediction)
        self._print_accuracy(accuracy, verbose=verbose)
        return accuracy

    #######################################################################
    #                             saving methods                          #
    #######################################################################

    def save_history(self, filepath):
        history = {'training_loss': self.total_loss,
                    'test_accuracy': self.test_accuracy
                    }
        with open(filepath, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
