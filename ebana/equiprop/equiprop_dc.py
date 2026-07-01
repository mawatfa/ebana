import numpy as np

from ..data_collection.process_data_accumulator import ProcessDataAccumulator
from ..simulation import Simulator

FREE, NUDGE = "free", "nudge"


class EqPropDC:
    def __init__(self, model):
        self.model = model
        self.sim = Simulator()

    def free_phase(self, X, idx, training=True):
        self.sim.load(self.model.build_netlist(X, idx))
        self.sim.solve()
        if training:
            self.measure(FREE)
        return {layer.name: layer.read(self.sim) for layer in self.model.outputs}

    def nudge_phase(self, forces):
        for layer in self.model.outputs:
            self.sim.alter(*layer.nudge(forces[layer.name]))
        self.sim.solve()
        self.measure(NUDGE)

    def measure(self, phase):
        self.measurements[phase] = {p.full_name: p.measure(self.sim) for p in self.model.trainable_parameters()}

    def train(self, X, Y, batch_size, process_num, n_processes):
        params = self.model.trainable_parameters()
        for p in params:
            p.grad = np.zeros(p.shape)
        self.measurements = {}
        recorder = ProcessDataAccumulator(self.model, batch_size)
        losses = {k: 0.0 for k in self.model.output_nodes}
        record_forces = self.model.monitor.save_injected_currents
        forces_seen = {k: np.zeros((batch_size, v)) for k, v in self.model.output_nodes.items()} if record_forces else {}
        beta = self.model.optimizer.beta

        for idx in range(process_num, batch_size, n_processes):
            outputs = self.free_phase(X, idx)
            recorder.save_metrics_during_training(idx, FREE, self.sim)

            forces = {}
            for name, y in outputs.items():
                loss, grad = self.model.loss[name](y, Y[name][idx])
                losses[name] += loss
                forces[name] = -beta * grad
                if record_forces:
                    forces_seen[name][idx] = forces[name]

            self.nudge_phase(forces)
            recorder.save_metrics_during_training(idx, NUDGE, self.sim)
            for p in params:
                p.grad += p.gradient(self.measurements[FREE][p.full_name], self.measurements[NUDGE][p.full_name])

        return {
            "process_layer_collected_data": recorder.get_data(),
            "process_parameter_gradients": {p.full_name: p.grad for p in params},
            "process_injected_currents": forces_seen,
            "process_losses": losses,
        }

    def infer(self, X, n_samples, process_num, n_processes):
        # zero for the samples of the other processes
        outputs = {layer.name: np.zeros((n_samples, layer.units)) for layer in self.model.outputs}
        for idx in range(process_num, n_samples, n_processes):
            for name, y in self.free_phase(X, idx, training=False).items():
                outputs[name][idx] = y
        return outputs
