import numpy as np
from .datacollector import DataCollector

class DataAccumulator:
    def __init__(self, model):
        self.model = model
        self.output_nodes = self.model.output_nodes
        self.dataset_size = self.model.dataset_size
        self.batch_size = self.model.batch_size
        self.num_batches = self.model.num_batches
        self.num_epochs = self.model.epochs
        self.n_processes_train = self.model.n_processes_train
        self.__initialize_data_collection()
        self.__initialize_injected_current()
        # mean loss per sample: {output: (epochs, batches)}
        self.total_loss = {k: np.zeros((self.num_epochs, self.num_batches)) for k in self.output_nodes}
        # validation results: {"loss" or metric name: {output: (epochs,)}}, NaN where not run
        self.validation = {}

    def get_training_loss(self):
        """Mean loss per sample for each epoch: {output: (epochs,)}."""
        return {k: v.mean(axis=1) for k, v in self.total_loss.items()}

    def get_validation(self, key="loss"):
        """Validation `key` ("loss" or a metric name) for each epoch: {output: (epochs,)}."""
        return self.validation[key]

    def store_validation(self, results, epoch_num):
        for key, per_output in results.items():
            history = self.validation.setdefault(key, {k: np.full(self.num_epochs, np.nan) for k in per_output})
            for k, value in per_output.items():
                history[k][epoch_num] = value

    def __initialize_data_collection(self):
        self.collector = DataCollector(self.batch_size, self.num_batches, self.num_epochs)
        for layer in self.model.layers:
            save_phase_data = self.model.monitor.save_phase_data
            save_batch_data = self.model.monitor.save_batch_data
            self.collector.add_layer_specs(layer, save_phase_data, save_batch_data)

    # TODO: This needs to be done automatically, just like the rest of the data that gets saved!
    def __initialize_injected_current(self):
        self.injected_currents = {}
        if self.model.monitor.save_injected_currents:
            for k, v in self.output_nodes.items():
                self.injected_currents[k] = {}
                for epoch in range(self.num_epochs):
                    self.injected_currents[k][epoch] = {}
                    for batch in range(self.num_batches):
                        self.injected_currents[k][epoch][batch] = np.zeros(shape=(self.batch_size, v))

    def save_variables_not_needing_aggregation(self, results, epoch_num, batch_num):
        for process_num in range(self.n_processes_train):

            for layer in self.model.layers:

                # Save per phase data
                if layer.save_sim_data:
                    if self.model.monitor.save_phase_data:
                        layer_data = results[process_num]["process_layer_collected_data"][layer.name]
                        for phase_group in layer_data.keys():
                            for data_key in layer_data[phase_group].keys():
                                for phase in layer_data[phase_group][data_key].keys():
                                    data = layer_data[phase_group][data_key][phase]
                                    self.collector.store_phase_data(layer.name, phase_group, data_key, epoch_num, batch_num, phase, data)

            if self.model.monitor.save_injected_currents:
                for k, _ in self.output_nodes.items():
                    self.injected_currents[k][epoch_num][batch_num] += results[process_num]["process_injected_currents"][k]


        # Save per batch data
        for layer in self.model.layers:
            if self.model.monitor.save_batch_data:
                if layer.save_sim_data:
                    for key, p in layer.params.items():
                        self.collector.store_batch_data(layer.name, key, epoch_num, batch_num, p.value)


    def _save_variables_needing_aggregation(self, results, epoch_num, batch_num):
        process_losses = {}

        for process_num in range(self.n_processes_train):

            # (a). Gradients: sum over all samples of the batch.
            for p in self.model.trainable_parameters():
                if process_num == 0:
                    p.grad = np.zeros(p.shape)
                p.grad += results[process_num]["process_parameter_gradients"][p.full_name]

            # (b). Losses (sums over each process's samples)
            for k in self.output_nodes:
                process_losses[k] = process_losses.get(k, 0.0) + results[process_num]["process_losses"][k]

        for k in self.output_nodes:
            self.total_loss[k][epoch_num][batch_num] = process_losses[k] / self.batch_size


    def get_data(self):
        return self.collector.collected_data
