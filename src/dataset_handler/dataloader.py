import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_size = dataset['output'].shape[0]
        self.output_size = dataset['output'].shape[1]

    def __len__(self):
        return len(self.dataset_size)

    def __iter__(self):
        return self

    def __next__(self):
        sample = {}
        sample['inputs'] = {}

        random_choice = np.random.choice(self.dataset_size, self.batch_size, replace=False)

        for k, v in self.dataset['inputs'].items():
            sample['inputs'][k] = v[random_choice]

        sample['output'] = self.dataset['output'][random_choice]

        return sample
