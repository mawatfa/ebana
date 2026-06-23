# imports <<<
import numpy as np
import multiprocessing

# the workers inherit the model by forking
Pool = multiprocessing.get_context("fork").Pool

from ..dataset_handler import Dataset, DataLoader
from ..equiprop import EqProp
from ..spicesimulation import Spice
from ..data_collection.data_accumulator import DataAccumulator
from ..stat_reporting.accuracy import AccuracyReporter
from ..stat_reporting.loss import LossReporter
from ..stat_reporting.validation import TrainingValidator
from ..data_storage.save_load_history import HistoryLoader
from .evaluator import Evaluator
# >>>

class Model:

    # constructor <<<
    def __init__(self, inputs, outputs, n_processes_train=1, n_processes_test=1):
        """
        The model is a built as a graph, with some nodes designated as inputs, and others designated as outputs
        """
        self.inputs = inputs                        # list of inputs to the network
        self.outputs = outputs                      # list of outputs of the network
        self.n_processes_train = n_processes_train  # number of parallel processes
        self.n_processes_test = n_processes_test    # number of parallel processes
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

    # write netlist <<<
    def write_netlist(self, dataset, filename, sample=0):
        Spice(self).write_netlist(filename, dataset.inputs, sample)
    # >>>

    # history saver/loader <<<
    def save_history(self, filepath):
        self.history_loader.save_history(filepath)

    def load_history(self, filepath):
        self.history_loader.load_history(filepath)
    # >>>

    # fit method <<<
    def fit(
        self,
        *,
        train_data: DataLoader,
        epochs: int,
        optimizer,
        metrics,
        loss_fn,
        validation_data: Dataset | None = None,
        initial_epoch: int = 0,
    ):
        """To resume, load the optimizer state and start at initial_epoch."""
        if not 0 <= initial_epoch < epochs:
            raise ValueError(f"need 0 <= initial_epoch < epochs, got {initial_epoch} and {epochs}")
        self.fit_method_called = True
        self.dataset_size = len(train_data.dataset)
        self.batch_size = train_data.batch_size
        self.output_nodes = train_data.dataset.output_sizes
        self.num_batches = len(train_data)
        self.train_data = train_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss_fn = loss_fn

        self.accumulator = DataAccumulator(model=self)
        self.accuracy_reporter = AccuracyReporter(model=self)
        self.loss_reporter = LossReporter(model=self, total_loss=self.accumulator.total_loss)
        self.training_validator = TrainingValidator(model=self, test_accuracy=self.accumulator.test_accuracy, metrics=self.metrics)
        self.history_loader = HistoryLoader(self.accumulator)

        for epoch_num in range(initial_epoch, self.epochs):
            self.epoch_num = epoch_num

            for batch_num, training_batch in enumerate(train_data.batches(epoch_num)):
                self.batch_num = batch_num + 1

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
                if self.validation_data is not None:
                    self.training_validator.evaluate_validation(self.validation_data, epoch_num, batch_num)
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

