import numpy as np

class DataSet:
    def __init__(self, inputs: dict = {}, outputs: dict = {}):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.get_size()

    def get_size(self):
        if not self.inputs:
            raise ValueError("Inputs are not defined")

        # Find the first non-empty array in the inputs
        for _, value in self.inputs.items():
            if value is not None:
                return value.shape[0]

        raise ValueError("No valid data in inputs")


class BatchGenerator:
    def __init__(self, dataset: DataSet, batch_size: int = 128, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.initialize_sizes()

    def initialize_sizes(self) -> None:
        self.dataset_size = -1
        self.output_size = {}
        for k, v in self.dataset.outputs.items():
            self.output_size[k] = v.shape[1]
            if self.dataset_size == -1:
                self.dataset_size = v.shape[0]
        self.num_batches = int(np.floor(self.dataset_size / self.batch_size))
        self.batch_counter = self.num_batches

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        return self

    def __next__(self) -> DataSet:
        batch = DataSet(inputs={}, outputs={})

        if self.shuffle:

            # random_choice = np.random.choice(self.dataset_size, self.batch_size, replace=False)
            random_choice = np.arange(self.dataset_size)

            for k, v in self.dataset.inputs.items():
                batch.inputs[k] = v[random_choice]

            for k, v in self.dataset.outputs.items():
                batch.outputs[k] = v[random_choice]

        else:

            for k, v in self.dataset.inputs.items():
                batch.inputs[k] = v[self.index : self.index + self.batch_size]

            for k, v in self.dataset.outputs.items():
                batch.outputs[k] = v[self.index : self.index + self.batch_size]

            self.batch_counter -= 1
            self.index += self.batch_size

            if self.batch_counter == 0:
                self.batch_counter = self.num_batches
                self.index = 0

        self.batch = batch

        return batch
