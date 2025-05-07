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
        self.__initialize_loss_and_accuracy()

    def get_training_loss(self):
        loss = {}
        for k in self.model.output_nodes.keys():
            lss = self.total_loss[k]
            if lss.ndim == 3:
                loss[k] = self.total_loss[k].sum(axis=1).sum(axis=1)
            elif lss.ndim == 2:
                loss[k] = self.total_loss[k].sum(axis=1)
            else:
                loss[k] = self.total_loss[k].sum()
            loss[k] /= self.model.batch_size
        return loss

    def get_test_accuracy(self):
        accuracy = {}
        for k in self.model.output_nodes.keys():
            acc = self.test_accuracy[k]
            if acc.ndim == 2:
                accuracy[k] = 100*self.test_accuracy[k].mean(axis=1)
            else:
                accuracy[k] = 100*self.test_accuracy[k]
        return accuracy

    def __initialize_data_collection(self):
        self.collector = DataCollector(self.batch_size, self.num_batches, self.num_epochs)
        for layer in self.model.computation_graph:
            save_phase_data = self.model.metrics.save_phase_data
            save_batch_data = self.model.metrics.save_batch_data
            self.collector.add_layer_specs(layer, save_phase_data, save_batch_data)

    # TODO: This needs to be done automatically, just like the rest of the data that gets saved!
    def __initialize_injected_current(self):
        self.injected_currents = {}
        if self.model.metrics.save_injected_currents:
            for k, v in self.output_nodes.items():
                self.injected_currents[k] = {}
                for epoch in range(self.num_epochs):
                    self.injected_currents[k][epoch] = {}
                    for batch in range(self.num_batches):
                        self.injected_currents[k][epoch][batch] = np.zeros(shape=(self.batch_size, v))

    def __initialize_loss_and_accuracy(self):
        # variables to save loss and accuracy
        self.total_loss = {}
        self.test_accuracy = {}

        if self.batch_size != self.dataset_size:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(shape=(self.num_epochs, self.num_batches, node_size))
                self.test_accuracy[node_name] = np.zeros(shape=(self.num_epochs, self.num_batches))
        else:
            for node_name, node_size in self.output_nodes.items():
                self.total_loss[node_name] = np.zeros(shape=(self.num_epochs, node_size))
                self.test_accuracy[node_name] = np.zeros(shape=(self.num_epochs,))


    def save_variables_not_needing_aggregation(self, results, epoch_num, batch_num):
        for process_num in range(self.n_processes_train):

            for layer in self.model.computation_graph:

                # Save per phase data
                if layer.save_sim_data:
                    if self.model.metrics.save_phase_data:
                        layer_data = results[process_num]["process_layer_collected_data"][layer.name]
                        for phase_group in layer_data.keys():
                            for data_key in layer_data[phase_group].keys():
                                for phase in layer_data[phase_group][data_key].keys():
                                    data = layer_data[phase_group][data_key][phase]
                                    self.collector.store_phase_data(layer.name, phase_group, data_key, epoch_num, batch_num, phase, data)

            if self.model.metrics.save_injected_currents:
                for k, _ in self.output_nodes.items():
                    self.injected_currents[k][epoch_num][batch_num] += results[process_num]["process_injected_currents"][k]


        # Save per batch data
        for layer in self.model.computation_graph:
            if self.model.metrics.save_batch_data:
                for key in layer.get_batch_data_spec().keys():
                    self.collector.store_batch_data(layer.name, key, epoch_num, batch_num, layer.get_batch_data()[key])


    def _save_variables_needing_aggregation(self, results, epoch_num, batch_num):
        process_losses = {}

        for process_num in range(self.n_processes_train):

            # (a). Voltage Drops: We need to add the voltage drops of all samples in a batch.
            for layer in self.model.computation_graph:
                if layer.trainable:
                    if process_num == 0:
                        layer.accumulated_gradient = np.zeros(shape=layer.shape)
                    layer.accumulated_gradient += results[process_num]["process_layer_accumulated_gradient"][layer.name]

            # (b). Losses
            for k, v in self.output_nodes.items():
                if process_num == 0:
                    process_losses[k] = np.zeros(shape=(v,))
                process_losses[k] += results[process_num]["process_losses"][k]

        # After collecting the losses from a batch, we need to save them in the overall loss file
        if self.batch_size != self.dataset_size:
            for k, _ in self.output_nodes.items():
                self.total_loss[k][epoch_num][batch_num] = process_losses[k]
        else:
            for k, _ in self.output_nodes.items():
                self.total_loss[k][epoch_num] = process_losses[k]


    def get_data(self):
        return self.collector.collected_data
