"""Worker processes. Each one keeps a copy of the model and simulates a share of the samples."""

import multiprocessing
from functools import partial

import numpy as np

from .algorithms.base import SampleCircuit, free_outputs
from .data_collection.process_data_accumulator import ProcessDataAccumulator
from .simulation import Simulator

_model = None   # this process's copy of the model


def _start(model):
    global _model
    _model = model


def _samples(values, X):
    # the netlist is loaded once, then the layers alter it for every sample
    for p in _model.parameters():
        p.value = values[p.full_name]
    sim = Simulator()
    sim.load(_model.build_netlist(X, 0))
    for i in range(len(next(iter(X.values())))):
        for layer in _model.layers:
            layer.set_sample(sim, X, i)
        yield i, sim


def _train(values, X, Y, indices, batch_size):
    model = _model
    grads = {p.full_name: np.zeros(p.shape) for p in model.trainable_parameters()}
    losses = {name: 0.0 for name in Y}
    forces = {}
    recorder = ProcessDataAccumulator(model, batch_size)
    for i, sim in _samples(values, X):
        circuit = SampleCircuit(model, sim, partial(recorder.save_metrics_during_training, indices[i]))
        result = model.algorithm.train_step(circuit, {k: v[i] for k, v in Y.items()}, model.loss)
        circuit.restore()
        for name, g in result.grads.items():
            grads[name] += g
        for name, value in result.losses.items():
            losses[name] += value
        forces[indices[i]] = result.forces
    return {"grads": grads, "losses": losses, "forces": forces, "recorded": recorder.get_data()}


def _infer(values, X, indices):
    outputs, algorithm = {}, _model.algorithm
    for i, sim in _samples(values, X):
        circuit = SampleCircuit(_model, sim)
        outputs[indices[i]] = algorithm.predict_step(circuit) if algorithm is not None else free_outputs(circuit)
        circuit.restore()
    return outputs


class Workers:
    """
    `n` processes, each holding a copy of `model` (n=0: run in this process).
    Every task carries the current parameter values, so the copies never go
    stale. The model is pickled, so hooks it holds (grad_func, update_func,
    schedules...) must be picklable: module-level functions, not lambdas.
    """

    def __init__(self, model, n):
        self.model, self.n = model, n
        if n:
            self.pool = multiprocessing.get_context().Pool(n, _start, (model,))
        else:
            self.pool = None
            _start(model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if self.pool is None:
            return
        if exc_type is None:
            self.pool.close()
        else:
            self.pool.terminate()
        self.pool.join()

    def _map(self, task, n_samples, data, *extra):
        values = {p.full_name: p.value for p in self.model.parameters()}
        chunks = [idx for idx in np.array_split(np.arange(n_samples), max(self.n, 1)) if len(idx)]
        args = [(values, *({k: v[idx] for k, v in d.items()} for d in data), idx.tolist(), *extra) for idx in chunks]
        if self.pool is None:
            return [task(*a) for a in args]
        return self.pool.starmap(task, args)

    def train(self, X, Y):
        n = len(next(iter(Y.values())))
        return self._map(_train, n, (X, Y), n)

    def infer(self, X, n_samples):
        per_sample = {}
        for part in self._map(_infer, n_samples, (X,)):
            per_sample.update(part)
        return {name: np.array([per_sample[i][name] for i in range(n_samples)])
                for name in per_sample[0]}
