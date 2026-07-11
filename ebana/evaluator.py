import numpy as np


def sample_losses(outputs, targets, loss):
    return {name: np.array([loss[name](y[i], targets[name][i])[0] for i in range(len(y))])
            for name, y in outputs.items()}


def summary(losses, outputs, targets, metrics):
    """Means over the samples: "loss/<output>", "<metric>/<output>", and "loss", the sum over the outputs."""
    results = {"loss": float(sum(np.mean(v) for v in losses.values()))}
    for name, y in outputs.items():
        results[f"loss/{name}"] = float(np.mean(losses[name]))
        for metric in metrics[name]:
            results[f"{metric.name}/{name}"] = float(np.mean(metric(y, targets[name])))
    return results
