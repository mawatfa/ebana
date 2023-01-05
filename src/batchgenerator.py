#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                generate sample batches from dataset                 #
#######################################################################

class BatchGenerator:

    def __init__(self, dataset, batch_size=128, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0

        self.output_size = {}
        for k, v in self.dataset['outputs'].items():
            self.output_size[k] = v.shape[1]
            self.dataset_size = v.shape[0]

        self.num_batches = int(np.floor(self.dataset_size / self.batch_size))
        self.batch_counter = self.num_batches

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self

    def __next__(self):
        batch = {}
        batch['inputs'] = {}
        batch['outputs'] = {}

        if self.shuffle:

            random_choice = np.random.choice(self.dataset_size, self.batch_size, replace=False)

            for k, v in self.dataset['inputs'].items():
                batch['inputs'][k] = v[random_choice]

            for k, v in self.dataset['outputs'].items():
                batch['outputs'][k] = v[random_choice]

        else:

            for k, v in self.dataset['inputs'].items():
                batch['inputs'][k] = v[self.index : self.index + self.batch_size]

            for k, v in self.dataset['outputs'].items():
                batch['outputs'][k] = v[self.index : self.index + self.batch_size]

            self.batch_counter -= 1
            self.index += self.batch_size

            if self.batch_counter == 0:
                self.batch_counter = self.num_batches
                self.index = 0

        self.batch = batch

        return batch

    def update_batch(self, optimizer):
        for layer in optimizer.model.computation_graph:
            if layer.layer_type in ['input_voltage_layer']:
                self.dataset['inputs'][layer.name][self.index - self.batch_size] -= 10000 * layer.voltage_drops
