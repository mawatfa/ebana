# imports <<<
import numpy as np
from multiprocessing import Pool

from ..dataset_handler import BatchGenerator
from ..equiprop import EqProp
from ..spicesimulation import Spice
from ..data_collection.data_accumulator import DataAccumulator
from ..stat_reporting.accuracy import AccuracyReporter
from ..stat_reporting.loss import LossReporter
from ..stat_reporting.validation import TrainingValidator
from ..data_storage.save_load_history import HistoryLoader
from .evaluator import Evaluator
# >>>

np.random.seed(111)

class Model:

    # constructor <<<
    def __init__(self, inputs, outputs, n_processes_train=1, n_processes_test=1, simulator='ngspice', simulation_kind='op'):
        """
        The model is a built as a graph, with some nodes designated as inputs, and others designated as outputs
        """
        self.inputs = inputs                        # list of inputs to the network
        self.outputs = outputs                      # list of outputs of the network
        self.n_processes_train = n_processes_train  # number of parallel processes
        self.n_processes_test = n_processes_test    # number of parallel processes
        self.set_simulator(simulator)
        self.set_simulation_type(simulation_kind)
        self.set_stopping_condition_loss(func=None)
        self.set_stopping_condition_accuracy(func=None)
        self.adaptive_learning = None
        self.stopping_condition_loss = None
        self.stopping_condition_accuracy = None
        self._build_computation_graph(inputs)
        self.fit_method_called = False
    # >>>

    # buld graph <<<
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
    # >>>

    # simulator methods <<<
    def set_simulator(self, simulator):
        self.simulator = 'ngspice'

    def set_simulation_type(self, simulation_kind, sampling_time=None):
        self.simulation_kind = simulation_kind
        self.sampling_time = sampling_time
    # >>>

    # adaptive learning methods <<<
    def set_adaptive_learning(self, func):
        self.adaptive_learning = True
        self.adaptive_learning_func = func

    def set_stopping_condition_loss(self, func):
        self.stopping_condition_loss = True
        self.stopping_condition_loss_func = func

    def set_stopping_condition_accuracy(self, func):
        self.stopping_condition_accuracy = True
        self.stopping_condition_accuracy_func = func
    # >>>

    # batch skipping <<<
    def _handle_batch_skipping(self, custom_begin: None | dict, dataloader: BatchGenerator) -> tuple[int, int]:
        if custom_begin is not None:
            custom_begin_epoch = custom_begin["epoch"]
            custom_begin_batch = custom_begin["batch"]
            self.load_history(custom_begin["history"])

            for epoch_num in range(1, self.epochs):
                for batch_num in range(1, self.num_batches):
                    if epoch_num != custom_begin_epoch or self.batch_num != custom_begin_batch:
                        next(dataloader)
                    else:
                        return epoch_num, batch_num
        return 0, 0
    # >>>

    # write netlist <<<
    def write_netlist(self, dataloader: BatchGenerator, filename: "str") -> None:
        spice = Spice(self)
        training_batch = next(dataloader)
        X = training_batch.inputs
        spice.write_netlist(filename, X)
    # >>>

    # history saver/loader <<<
    def save_history(self, filepath):
        self.history_loader.save_history(filepath)

    def load_history(self, filepath):
        self.history_loader.load_history(filepath)
    # >>>

    # fit method <<<
    def fit(self, *, train_dataloader, epochs, optimizer, metrics, loss_fn, test_dataloader=None, custom_begin=None):
        self.fit_method_called = True
        self.dataset_size = train_dataloader.dataset_size
        self.batch_size = train_dataloader.batch_size
        self.output_nodes = train_dataloader.output_size
        self.num_batches = train_dataloader.num_batches
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss_fn = loss_fn

        self.accumulator = DataAccumulator(model=self)
        self.accuracy_reporter = AccuracyReporter(model=self)
        self.loss_reporter = LossReporter(model=self, total_loss=self.accumulator.total_loss)
        self.training_validator = TrainingValidator(model=self, test_accuracy=self.accumulator.test_accuracy, metrics=self.metrics)
        self.history_loader = HistoryLoader(self.accumulator)

        begin_epoch_num, begin_batch_num = self._handle_batch_skipping(custom_begin, self.train_dataloader)
        end_training = False

        for epoch_num in range(begin_epoch_num, self.epochs):
            self.epoch_num = epoch_num

            for batch_num in range(begin_batch_num, self.num_batches):
                self.batch_num = batch_num + 1

                # Get batch of samples
                training_batch = next(self.train_dataloader)
                # type: dict { key = input layer name, value = 2D numpy array of inputs }
                X = training_batch.inputs
                Y = training_batch.outputs

                #######################################################################
                #                           Multiprocessing                           #
                #######################################################################
                # Prepare arguments for the processes
                child_processes_args = []
                for process_num in range(self.n_processes_train):
                    child_processes_args.append((X, Y, self.batch_size, process_num, self.n_processes_train))

                # Run processes and get result
                with Pool(processes=self.n_processes_train) as pool:
                    results = pool.starmap(EqProp(self).train, child_processes_args)

                #######################################################################
                #                 Aggregate result from processes                     #
                #######################################################################
                # The `results` variable is a list of `self.n_processes_train` elements.
                # Each element is a dictionary.

                ##  1. Some of the elements from the dictionary such as voltag_drops and losses have to be aggregated.
                self.accumulator._save_variables_needing_aggregation(results, epoch_num, batch_num)

                ##  2. Other elements require no aggregation.
                self.accumulator.save_variables_not_needing_aggregation(results, epoch_num, batch_num)

                #######################################################################
                #                          Batch Post Processing                      #
                #######################################################################
                # Optimize training parameters
                optimizer.step()

                # Print training loss per iteration
                if self.metrics.verbose:
                    self.loss_reporter.print_iteration_loss(epoch_num, batch_num)
                    self.loss_reporter.print_total_loss()

                # Validate on test/validation dataset
                if self.test_dataloader:
                    self.training_validator.evaluate_validation(self.test_dataloader, epoch_num, batch_num)

                    if self.stopping_condition_accuracy:
                        stop_condition = self.stopping_condition_accuracy_func(batch_num, epoch_num, self.accumulator.test_accuracy)
                        if stop_condition:
                            end_training = True
                            break

                if self.stopping_condition_loss:
                    stop_condition = self.stopping_condition_loss_func(batch_num, epoch_num, self.accumulator.total_loss)
                    if stop_condition:
                        end_training = True
                        break

                # I need to move this one indentation back so that it is only executed after the end of an epoch, not a batch
                if self.adaptive_learning:
                    self.adaptive_learning_func()

            # self.save_history("history.pickle")

            if end_training:
                break
    # >>>

    # evaluate method <<<
    def evaluate(self, dataset, loss_fn=None, verbose=True):
        """
        Evaluate dataset and return accuracy.
        """
        evaluator = Evaluator(self, loss_fn=loss_fn)
        return evaluator.evaluate(dataset, verbose=verbose)
    # >>>

    # predict method <<<
    def predict(self, dataset, loss_fn=None, verbose=True):
        """
        Evaluate the network on the dataset and return the node voltages,
        as well as the network predicton.
        """
        predictor = Evaluator(self, loss_fn=loss_fn)
        return predictor.predict(dataset)
    # >>>

