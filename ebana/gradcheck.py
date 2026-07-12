import numpy as np

from .algorithms.base import SampleCircuit
from .simulation import Simulator


def _loss(model, sim, inputs, index, targets, loss):
    sim.load(model.build_netlist(inputs, index))
    sim.solve()
    return sum(loss[layer.name](layer.read(sim), targets[layer.name])[0] for layer in model.outputs)


def gradient_check(model, dataset, loss, algorithm, index=0, parameters=None, elements=None, step=1e-4, seed=0):
    """
    Compares the gradient the algorithm finds for one sample with central
    differences of the loss, (L(p + h) - L(p - h)) / 2h, where h is `step`
    times |p|. `elements` picks that many random elements of each parameter.
    Returns {name: {"index", "estimate", "numerical"}}.
    """
    loss, _ = model._per_output(loss, None)
    targets = {name: v[index] for name, v in dataset.outputs.items()}
    params = [model.parameter(n) for n in parameters] if parameters else model.trainable_parameters()
    rng = np.random.default_rng(seed)

    was_trainable = {p.full_name: p.trainable for p in model.parameters()}
    for p in model.parameters():
        p.trainable = p in params
    try:
        sim = Simulator()
        sim.load(model.build_netlist(dataset.inputs, index))
        estimate = algorithm.train_step(SampleCircuit(model, sim), targets, loss).grads
    finally:
        for p in model.parameters():
            p.trainable = was_trainable[p.full_name]

    results = {}
    for p in params:
        chosen = list(np.ndindex(p.shape))
        if elements is not None and elements < len(chosen):
            chosen = [chosen[i] for i in rng.choice(len(chosen), elements, replace=False)]
        numerical = []
        original = p.value.copy()
        try:
            for idx in chosen:
                h = step * abs(original[idx]) if original[idx] != 0 else step
                values = []
                for sign in (1, -1):
                    p.value = original.copy()
                    p.value[idx] += sign * h
                    values.append(_loss(model, sim, dataset.inputs, index, targets, loss))
                numerical.append((values[0] - values[1]) / (2 * h))
        finally:
            p.value = original
        results[p.full_name] = {
            "index": chosen,
            "estimate": np.array([estimate[p.full_name][idx] for idx in chosen]),
            "numerical": np.array(numerical),
        }
    return results
