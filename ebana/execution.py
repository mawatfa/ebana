"""Worker processes. Each one keeps a copy of the model and simulates a share of the samples."""

import multiprocessing
from types import SimpleNamespace

import numpy as np

from .algorithms import SampleCircuit, free_outputs
from .simulation import Simulator

_job = None


def _start(model, algorithm, loss, probes):
    global _job
    _job = SimpleNamespace(model=model, algorithm=algorithm, loss=loss, probes=probes)


def _samples(values, X):
    # the netlist is loaded once, then the layers alter it for every sample
    model = _job.model
    for p in model.parameters():
        p.value = values[p.full_name]
    sim = Simulator()
    sim.load(model.build_netlist(X, 0))
    for i in range(len(next(iter(X.values())))):
        for layer in model.layers:
            layer.set_sample(sim, X, i)
        yield i, sim


def _train(values, X, Y, positions, probe_names):
    model, probes = _job.model, {name: _job.probes[name] for name in probe_names}
    grads = {p.full_name: np.zeros(p.shape) for p in model.trainable_parameters()}
    samples = {}
    for i, sim in _samples(values, X):
        measured = {name: {} for name in probes}

        def on_solve(phase, sim):
            for name, probe in probes.items():
                measured[name][phase] = probe(model, sim)

        circuit = SampleCircuit(model, sim, on_solve if probes else None)
        result = _job.algorithm.train_step(circuit, {k: v[i] for k, v in Y.items()}, _job.loss)
        circuit.restore()
        for name, g in result.grads.items():
            grads[name] += g
        samples[positions[i]] = {"loss": result.losses, "output": result.outputs,
                                 "force": result.forces, "probes": measured}
    return {"grads": grads, "samples": samples}


def _infer(values, X, positions):
    outputs, algorithm = {}, _job.algorithm
    for i, sim in _samples(values, X):
        circuit = SampleCircuit(_job.model, sim)
        outputs[positions[i]] = algorithm.predict_step(circuit) if algorithm is not None else free_outputs(circuit)
        circuit.restore()
    return outputs


class Workers:
    """
    n processes (0 runs everything in this process). Every task sends the
    current parameter values. The model, algorithm, loss and probes are
    pickled, so functions they hold must be defined at module level.
    """

    def __init__(self, model, n, algorithm=None, loss=None, probes=None):
        self.model, self.n = model, n
        args = (model, algorithm, loss, probes or {})
        if n:
            self.pool = multiprocessing.get_context().Pool(n, _start, args)
        else:
            self.pool = None
            _start(*args)

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

    def train(self, X, Y, probe_names=()):
        """The gradients summed over the batch, and the results of each sample in batch order."""
        n = len(next(iter(Y.values())))
        parts = self._map(_train, n, (X, Y), list(probe_names))
        grads = {name: sum(part["grads"][name] for part in parts) for name in parts[0]["grads"]}
        samples = {}
        for part in parts:
            samples.update(part["samples"])
        return grads, [samples[i] for i in range(n)]

    def infer(self, X, n_samples):
        per_sample = {}
        for part in self._map(_infer, n_samples, (X,)):
            per_sample.update(part)
        return {name: np.array([per_sample[i][name] for i in range(n_samples)])
                for name in per_sample[0]}
