# imports <<<
import numpy as np
from multiprocessing import Pool
from ..equiprop import EqProp
from ..stat_reporting.accuracy import AccuracyReporter
# >>>

class Evaluator:
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.loss_fn = loss_fn
        self.accuracy_reporter = AccuracyReporter(model)

    def _init_evaluation_variables(self, dataset):
        """
        Initialize model variables when the evaluate method is called
        outside the fit method.
        """
        self.model.loss_fn = self.loss_fn
        self.model.output_nodes = {}
        for key, val in dataset.outputs.items():
            self.model.output_nodes[key] = val.shape[1]

    # evaluate method <<<
    def evaluate(self, dataset, verbose=True):
        """
        Get model prediction and measure the accuracy.
        """
        if not self.model.fit_method_called:
            self._init_evaluation_variables(dataset)

        results = self._simulate_circuit(dataset, mode="evaluating")
        prediction, losses = self._collect_outputs_evaluation(results)

        accuracy = self.accuracy_reporter.measure_accuracy(dataset, prediction)
        self.accuracy_reporter.print_accuracy(accuracy, verbose=verbose)

        return accuracy, losses

    def _collect_outputs_evaluation(self, results):
        process_losses = {}
        process_predictions = {}
        for process_num in range(self.model.n_processes_test):
            for k, v in self.model.output_nodes.items():
                if process_num == 0:
                    process_predictions[k] = np.zeros(shape=(self.dataset_size, v))
                    process_losses[k] = np.zeros(shape=(v,))
                process_predictions[k] += results[process_num]["process_predictions"][k]
                process_losses[k] += results[process_num]["process_losses"][k] / self.dataset_size
        return process_predictions, process_losses
    # >>>

    # predict method <<<
    def predict(self, dataset):
        """
        Get model prediction only.
        """
        if not self.model.fit_method_called:
            self._init_evaluation_variables(dataset)

        results = self._simulate_circuit(dataset, mode="predicting")
        output_voltages, predictions = self._collect_outputs_prediction(results)

        return output_voltages, predictions

    def _collect_outputs_prediction(self, results):
        process_output = {}
        process_prediction = {}
        for process_num in range(self.model.n_processes_test):
            for k, v in self.model.output_nodes.items():
                if process_num == 0:
                    process_output[k] = np.zeros(shape=(self.dataset_size, v))
                    process_prediction[k] = np.zeros(shape=(self.dataset_size, v))
                process_output[k] += results[process_num]["process_output"][k]
                process_prediction[k] += results[process_num]["process_prediction"][k]
        return process_output, process_prediction

    # >>>

    # simulate circuit <<<
    def _simulate_circuit(self, dataset, mode):
        X = dataset.inputs
        Y = dataset.outputs

        self.dataset_size = dataset.get_size()

        # prepare arguments for the processes
        child_processes_args = []
        for _, _ in self.model.output_nodes.items():
            for process_num in range(self.model.n_processes_test):
                child_processes_args.append(
                    (X, Y, self.dataset_size, process_num, self.model.n_processes_test)
                )

        # run processes
        pool = Pool(processes=self.model.n_processes_test)

        if mode == "evaluating":
            results = pool.starmap(EqProp(self.model).evaluate, child_processes_args)
        else:
            results = pool.starmap(EqProp(self.model).predict, child_processes_args)

        return results
    # >>>
