"""What the worker processes run."""

import numpy as np

from .algorithms.base import SampleCircuit, free_outputs
from .data_collection.process_data_accumulator import ProcessDataAccumulator
from .simulation import Simulator


def _circuit(model, X, idx, on_solve=None):
    sim = Simulator()
    sim.load(model.build_netlist(X, idx))
    return SampleCircuit(model, sim, on_solve)


def train_samples(model, X, Y, batch_size, process_num, n_processes):
    algorithm, loss = model.algorithm, model.loss
    grads = {p.full_name: np.zeros(p.shape) for p in model.trainable_parameters()}
    losses = {name: 0.0 for name in model.output_nodes}
    recorder = ProcessDataAccumulator(model, batch_size)
    keep_forces = model.monitor.save_injected_currents
    forces = {k: np.zeros((batch_size, v)) for k, v in model.output_nodes.items()} if keep_forces else {}

    for idx in range(process_num, batch_size, n_processes):
        record = lambda phase, sim, idx=idx: recorder.save_metrics_during_training(idx, phase, sim)
        result = algorithm.train_step(_circuit(model, X, idx, record), {k: v[idx] for k, v in Y.items()}, loss)
        for name, g in result.grads.items():
            grads[name] += g
        for name, value in result.losses.items():
            losses[name] += value
        if keep_forces:
            for name, f in result.forces.items():
                forces[name][idx] = f

    return {
        "process_layer_collected_data": recorder.get_data(),
        "process_parameter_gradients": grads,
        "process_injected_currents": forces,
        "process_losses": losses,
    }


def infer_samples(model, X, n_samples, process_num, n_processes):
    # zero for the samples of the other processes
    algorithm = model.algorithm
    outputs = {layer.name: np.zeros((n_samples, layer.units)) for layer in model.outputs}
    for idx in range(process_num, n_samples, n_processes):
        circuit = _circuit(model, X, idx)
        values = algorithm.predict_step(circuit) if algorithm is not None else free_outputs(circuit)
        for name, y in values.items():
            outputs[name][idx] = y
    return outputs
