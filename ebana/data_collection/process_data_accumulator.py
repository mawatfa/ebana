from .datacollector import DataCollector

class ProcessDataAccumulator:
    def __init__(self, model, batch_size):
        self.model = model
        self.collector = DataCollector(batch_size)
        for layer in self.model.computation_graph:
            self.collector.add_layer_specs(layer, phase_specs=True, batch_specs=False)

    def save_metrics_during_training(self, idx, phase):
        for layer in self.model.computation_graph:
            if layer.save_sim_data:
                for phase_group, phase_data_spec in layer.get_phase_data_spec().items():
                    for data_key in phase_data_spec.keys():
                        self.collector.store_phase_data(
                            layer_name=layer.name,
                            phase_group=phase_group,
                            data_key=data_key,
                            epoch=None,
                            batch=idx,
                            phase=phase,
                            data=layer.get_phase_data()[phase_group][phase][data_key]
                        )

    def get_data(self):
        return self.collector.collected_data
