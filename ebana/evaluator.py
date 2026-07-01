import multiprocessing

import numpy as np

from .equiprop import EqProp

# the workers inherit the model by forking
Pool = multiprocessing.get_context("fork").Pool


def model_outputs(model, dataset):
    n = model.n_processes_test
    args = [(dataset.inputs, len(dataset), process_num, n) for process_num in range(n)]
    with Pool(processes=n) as pool:
        parts = pool.starmap(EqProp(model).infer, args)
    # each process filled only its own samples
    return {name: np.sum([part[name] for part in parts], axis=0) for name in parts[0]}


def evaluate(model, dataset, loss, metrics):
    """Mean loss and metrics: {"loss": {output: value}, metric name: {output: value}}."""
    outputs = model_outputs(model, dataset)
    results = {"loss": {}}
    for name, y in outputs.items():
        target = dataset.outputs[name]
        results["loss"][name] = float(np.mean([loss[name](y[i], target[i])[0] for i in range(len(y))]))
        for metric in metrics[name]:
            results.setdefault(metric.name, {})[name] = float(np.mean(metric(y, target)))
    return results
