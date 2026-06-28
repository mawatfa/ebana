from .datacollector import DataCollector

class ProcessDataAccumulator:
    def __init__(self, model, batch_size):
        self.model = model
        self.collector = DataCollector(batch_size)
        self.enabled = self.model.metrics.save_phase_data
        for layer in self.model.layers:
            self.collector.add_layer_specs(layer, phase_specs=self.enabled, batch_specs=False)

    def save_metrics_during_training(self, idx, phase, sim):
        if not self.enabled:
            return
        for layer in self.model.layers:
            if layer.save_sim_data:
                for phase_group, values in layer.record(sim).items():
                    for data_key, data in values.items():
                        self.collector.store_phase_data(
                            layer_name=layer.name,
                            phase_group=phase_group,
                            data_key=data_key,
                            epoch=None,
                            batch=idx,
                            phase=phase,
                            data=data,
                        )

    def get_data(self):
        return self.collector.collected_data
