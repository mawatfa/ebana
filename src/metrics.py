import numpy as np
import collections

class Metrics:
    def __init__(self, model, save_output_voltages=False, save_power_params=False, verbose=True, validate_every=None):
        self.model = model
        self.save_output_voltages = save_output_voltages
        self.save_power_params = save_power_params
        self.verbose = verbose
        self.validate_every = validate_every

        self.process_training_output_voltages = {}
        self.process_training_power_params = {}

        # TODO: I probably can use a simple dictionary instead of a defaultdict
        if save_output_voltages:
            self.output_voltages = collections.defaultdict(dict)

        if save_power_params:
            self.power_params = collections.defaultdict(dict)

    def __should_save(self, param):
        return param == "all" or (param == "last" and self.model.epoch_num == self.model.epochs - 1)

    def initialize_metrics_during_training(self, batch_size):
        should_save_output_voltages = self.__should_save(self.save_output_voltages)
        should_save_power_params = self.__should_save(self.save_power_params)


        for layer in self.model.computation_graph:

            if should_save_output_voltages:
                if layer.save_output_voltage:
                    self.process_training_output_voltages[layer.name] = np.zeros(shape=(batch_size, layer.output_shape[0]))

            if should_save_power_params:
                if layer.save_power_params:
                    self.process_training_power_params[layer.name] = np.zeros(shape=(batch_size, 4, layer.output_shape[0]))

    def save_metrics_during_training(self, idx, output_voltages=True, power_params=True, begin=0, end=2):
        should_save_output_voltages = self.__should_save(self.save_output_voltages)
        should_save_power_params = self.__should_save(self.save_power_params)

        for layer in self.model.computation_graph:

            if should_save_output_voltages and output_voltages:
                if layer.save_output_voltage:
                    self.process_training_output_voltages[layer.name][idx] = layer.output_voltages

            if should_save_power_params and power_params:
                if layer.save_power_params:
                    self.process_training_power_params[layer.name][idx][begin:end] = self.spice.get_power_parameters(layer.out_net_names, layer.current_net_names)

    def accumulate_metrics_from_processes(self, results, process_num):
        should_save_output_voltages = self.__should_save(self.save_output_voltages)
        should_save_power_params = self.__should_save(self.save_power_params)

        for layer in self.model.computation_graph:

            if should_save_output_voltages:
                if layer.save_output_voltage:
                    if process_num == 0:
                        self.output_voltages[self.model.epoch_num][layer.name] = np.zeros(shape=(self.batch_size, layer.output_shape[0]))
                    self.output_voltages[self.model.epoch_num][layer.name] += results[process_num]["process_layer_output_voltage"][layer.name]


            if should_save_power_params:
                if layer.save_power_params:
                    if process_num == 0:
                        self.power_params[self.model.epoch_num][layer.name] = np.zeros( shape=(self.batch_size, 4, layer.output_shape[0]))
                    self.power_params[self.model.epoch_num][layer.name] += results[process_num]["process_power_params"][layer.name]

    #######################################################################
    #                           metrics methods                           #
    #######################################################################

    def _evaluate_validation(self, dataset, epoch_num, batch_num):
        if "epoch_num" in self.validate_every.keys():
            if (
                (epoch_num % self.validate_every["epoch_num"] == 0)
                and (batch_num == self.model.num_batches - 1)
                or (epoch_num == self.model.epochs - 1)
            ):
                _, accuracy = self.model.evaluate(dataset, verbose=self.verbose)
                for k in self.model.output_nodes.keys():
                    self.model.test_accuracy[k][epoch_num] = accuracy[k]

        elif "batch_num" in self.validate_every.keys():
            if ((batch_num + 1) % self.validate_every["batch_num"] == 0) or (
                (epoch_num == self.model.epochs - 1) and (batch_num == self.model.num_batches)
            ):
                accuracy = self.model.evaluate(dataset, verbose=self.verbose)
                for k in self.model.output_nodes.keys():
                    self.model.test_accuracy[k][epoch_num][batch_num] = accuracy[k]

    def _measure_accuracy(self, dataset, prediction):
        target = dataset["outputs"]
        accuracy = {}
        for k in self.model.output_nodes.keys():
            c = self.model.loss_fn.verify_result(target[k], prediction[k])
            accuracy[k] = np.round(np.mean(c), 2)
        return accuracy

    def _print_accuracy(self, accuracy, verbose=False):
        if verbose:
            print("\n-->Accuracy<--")
            for k in self.model.output_nodes.keys():
                print(f"    {k}: {accuracy[k]*100:.2f}%")
            print("")

    def _print_iteration_loss(self, epoch_num, batch_num):
        """
        Print iteration loss

        @param iter_n: iteration we're currently in
        """
        loss, loss_mean = {}, {}

        if self.model.dataset_size != self.model.batch_size:
            print(f"Iteration {epoch_num}/{batch_num}:")
            for k in self.model.output_nodes.keys():
                loss[k] = self.model.total_loss[k][epoch_num][batch_num] / self.model.batch_size
                loss_mean[k] = np.mean(loss[k])
        else:
            print(f"Iteration {epoch_num}:")
            for k in self.model.output_nodes.keys():
                loss[k] = self.model.total_loss[k][epoch_num] / self.model.dataset_size
                loss_mean[k] = np.mean(loss[k])

        for k in self.model.output_nodes.keys():
            print(f"    {k}: {loss_mean[k]:.7f} -> {loss[k]}")
