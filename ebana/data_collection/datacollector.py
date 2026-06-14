import numpy as np

class DataCollector:
    def __init__(self, batch_size, num_batches=None, epochs=None, phases=["free", "nudge"]):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.epochs = epochs
        self.phases = phases
        self.collected_data = {}

        # If epochs and num_batches are specified, we store data using [epoch][batch][phase]
        # Otherwise, we store data using [phase] only
        self.training_mode = (epochs is not None) and (num_batches is not None)

    def add_layer_specs(self, layer, phase_specs=False, batch_specs=False):
        """
        Phase data spec (e.g., input/output voltages saved each phase)
        Batch data spec (e.g., weights saved once per batch)
        """
        if not layer.save_sim_data:
            return

        self.collected_data[layer.name] = {}

        # Phase data spec
        if phase_specs:
            phase_specs = layer.get_phase_data_spec()
            for phase_group, phase_data_spec in phase_specs.items():
                for data_key, shape in phase_data_spec.items():
                    self._reserve_phase_data(layer.name, phase_group, data_key, shape)

        if batch_specs:
            # Batch data spec
            batch_specs = layer.get_batch_data_spec()
            for data_key, shape in batch_specs.items():
                self._reserve_batch_data(layer.name, data_key, shape)

    def _reserve_phase_data(self, layer_name, phase_group, data_key, shape):
        """
        Create storage for data that is saved each phase.
        """
        # print(layer_name, phase_group, data_key, shape)
        # exit()

        if phase_group not in self.collected_data[layer_name]:
            self.collected_data[layer_name][phase_group] = {}
        if self.training_mode:
            self.collected_data[layer_name][phase_group][data_key] = {
                epoch: {
                    batch: {
                        phase: np.zeros((self.batch_size, *shape))
                        for phase in self.phases
                    }
                    for batch in range(self.num_batches)
                }
                for epoch in range(self.epochs)
            }
        else:
            self.collected_data[layer_name][phase_group][data_key] = {
                phase: np.zeros((self.batch_size, *shape))
                for phase in self.phases
            }

    def _reserve_batch_data(self, layer_name, data_key, shape):
        """
        Create storage for data that is saved once after each batch.
        """
        # Only applies in training mode
        if self.training_mode:
            self.collected_data[layer_name][data_key] = {
                epoch: {
                    batch: np.zeros(shape)
                    for batch in range(self.num_batches)
                }
                for epoch in range(self.epochs)
            }

    def store_phase_data(self, layer_name, phase_group, data_key, epoch, batch, phase, data):
        # If not training_mode, ignore epoch/batch
        if self.training_mode:
            self.collected_data[layer_name][phase_group][data_key][epoch][batch][phase] += data
        else:
            self.collected_data[layer_name][phase_group][data_key][phase][batch] += data

    def store_batch_data(self, layer_name, data_key, epoch, batch, data):
        # Only meaningful in training mode
        if self.training_mode:
            self.collected_data[layer_name][data_key][epoch][batch] += data

    def get_collected_data(self):
        return self.collected_data
