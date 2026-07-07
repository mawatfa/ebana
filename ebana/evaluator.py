import numpy as np


def evaluate(outputs, dataset, loss, metrics):
    """Mean loss and metrics: {"loss": {output: value}, metric name: {output: value}}."""
    results = {"loss": {}}
    for name, y in outputs.items():
        target = dataset.outputs[name]
        results["loss"][name] = float(np.mean([loss[name](y[i], target[i])[0] for i in range(len(y))]))
        for metric in metrics[name]:
            results.setdefault(metric.name, {})[name] = float(np.mean(metric(y, target)))
    return results
