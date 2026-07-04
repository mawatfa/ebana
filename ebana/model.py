"""The model: a circuit made of connected layers, and its training loop."""

# imports <<<
import multiprocessing

# the workers inherit the model by forking
Pool = multiprocessing.get_context("fork").Pool

from .dataset_handler import Dataset, DataLoader
from . import execution
from .simulation import Netlist
from .layers import Input
from .data_collection.data_accumulator import DataAccumulator
from .stat_reporting.loss import LossReporter
from .stat_reporting.validation import TrainingValidator
from .data_storage.save_load_history import HistoryLoader
from .monitor import Monitor
from . import evaluator
# >>>


def _connected_layers(outputs):
    """Every layer connected to the outputs, parents first."""
    seen, stack = {}, list(outputs)
    while stack:
        layer = stack.pop()
        if id(layer) not in seen:
            seen[id(layer)] = layer
            stack.extend(layer.parents + layer.children)
    return sorted(seen.values(), key=lambda layer: layer.index)


class Model:

    # constructor <<<
    def __init__(self, outputs, n_processes_train=1, n_processes_test=1, temperature=27):
        self.outputs = list(outputs)
        self.temperature = temperature
        self.n_processes_train = n_processes_train  # number of parallel processes
        self.n_processes_test = n_processes_test    # number of parallel processes
        self.layers = _connected_layers(self.outputs)
        self.inputs = [layer for layer in self.layers if isinstance(layer, Input)]
        names = [layer.name for layer in self.layers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"layer names must be unique: {duplicates}")
        self.fit_method_called = False
        self.algorithm = None
    # >>>

    # parameters <<<
    def parameters(self):
        return [p for layer in self.layers for p in layer.params.values()]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.trainable]
    # >>>

    def check_dataset(self, dataset, outputs=True):
        missing = [layer.name for layer in self.inputs if layer.name not in dataset.inputs]
        if missing:
            raise KeyError(f"the dataset has no inputs for {missing}")
        if outputs:
            names = {layer.name for layer in self.outputs}
            if set(dataset.outputs) != names:
                raise KeyError(f"dataset outputs {sorted(dataset.outputs)} don't match the output layers {sorted(names)}")
            for layer in self.outputs:
                if dataset.output_sizes[layer.name] != layer.units:
                    raise ValueError(f"output '{layer.name}' has {layer.units} units, the dataset has {dataset.output_sizes[layer.name]}")

    def _per_output(self, loss, metrics):
        """A loss or metrics given once are used for every output."""
        names = [layer.name for layer in self.outputs]
        if not isinstance(loss, dict):
            loss = {name: loss for name in names}
        if metrics is None:
            metrics = []
        if not isinstance(metrics, dict):
            metrics = {name: list(metrics) for name in names}
        for what, value in (("loss", loss), ("metrics", metrics)):
            if set(value) != set(names):
                raise KeyError(f"{what} given for {sorted(value)}, the outputs are {sorted(names)}")
        return loss, metrics

    # netlist <<<
    def build_netlist(self, inputs, sample_index=0):
        netlist = Netlist(temperature=self.temperature)
        for layer in self.layers:
            layer.stamp(netlist.block(layer.name), inputs, sample_index)
        return netlist

    def write_netlist(self, dataset, filename, sample=0):
        with open(filename, "w") as f:
            f.write(str(self.build_netlist(dataset.inputs, sample)))
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
        loss,
        algorithm,
        metrics=None,
        validation_data: Dataset | None = None,
        initial_epoch: int = 0,
        monitor: Monitor | None = None,
    ):
        """
        loss and metrics are used for every output, or given per output in a
        dict. To resume, load the optimizer state and start at initial_epoch.
        """
        if not 0 <= initial_epoch < epochs:
            raise ValueError(f"need 0 <= initial_epoch < epochs, got {initial_epoch} and {epochs}")
        self.check_dataset(train_data.dataset)
        if validation_data is not None:
            self.check_dataset(validation_data)
        self.fit_method_called = True
        self.dataset_size = len(train_data.dataset)
        self.batch_size = train_data.batch_size
        self.output_nodes = train_data.dataset.output_sizes
        self.num_batches = len(train_data)
        self.train_data = train_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.optimizer = optimizer
        self.algorithm = algorithm
        self.loss, self.metrics = self._per_output(loss, metrics)
        self.monitor = monitor or Monitor()

        self.accumulator = DataAccumulator(model=self)
        self.loss_reporter = LossReporter(model=self, accumulator=self.accumulator)
        self.training_validator = TrainingValidator(model=self, accumulator=self.accumulator, monitor=self.monitor)
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
                    results = pool.starmap(execution.train_samples, [(self, *args) for args in child_processes_args])

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

                if self.monitor.verbose:
                    self.loss_reporter.print_iteration_loss(epoch_num, batch_num)

                if self.validation_data is not None:
                    self.training_validator.evaluate_validation(self.validation_data, epoch_num, batch_num)

            if self.monitor.verbose:
                self.loss_reporter.print_epoch_loss(epoch_num)
    # >>>

    # evaluate and predict <<<
    def evaluate(self, dataset, loss=None, metrics=None, verbose=True):
        """Uses the loss and metrics of the last fit when none are given."""
        self.check_dataset(dataset)
        if loss is None:
            if not self.fit_method_called:
                raise ValueError("evaluate() needs a loss before the model has been fit")
            loss = self.loss
            metrics = self.metrics if metrics is None else metrics
        loss, metrics = self._per_output(loss, metrics)
        results = evaluator.evaluate(self, dataset, loss, metrics)
        if verbose:
            print("evaluation: " + "; ".join(
                f"{key} " + ", ".join(f"{k} {v:.6g}" for k, v in values.items()) for key, values in results.items()))
        return results

    def predict(self, dataset):
        self.check_dataset(dataset, outputs=False)
        return evaluator.model_outputs(self, dataset)
    # >>>
