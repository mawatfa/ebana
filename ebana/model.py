import numpy as np

from .execution import Workers
from .layers import Input
from .simulation import Netlist
from .training import Training, per_output, sample_losses, summary


def _connected_layers(outputs):
    """Every layer connected to the outputs, parents first."""
    seen, stack = {}, list(outputs)
    while stack:
        layer = stack.pop()
        if id(layer) not in seen:
            seen[id(layer)] = layer
            stack.extend(layer.parents + layer.children)
    return sorted(seen.values(), key=lambda layer: layer.index)


class Model:

    # constructor <<<
    def __init__(self, outputs, temperature=27):
        self.outputs = list(outputs)
        self.temperature = temperature
        self.layers = _connected_layers(self.outputs)
        self.inputs = [layer for layer in self.layers if isinstance(layer, Input)]
        names = [layer.name for layer in self.layers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"layer names must be unique: {duplicates}")
    # >>>

    # layers and parameters <<<
    def layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"no layer named '{name}'")

    def parameters(self):
        return [p for layer in self.layers for p in layer.params.values()]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.trainable]

    def parameter(self, full_name):
        for p in self.parameters():
            if p.full_name == full_name:
                return p
        raise KeyError(f"no parameter named '{full_name}'")

    def parameter_values(self):
        return {p.full_name: p.value.copy() for p in self.parameters()}

    def set_parameter_values(self, values):
        params = {p.full_name: p for p in self.parameters()}
        if set(values) != set(params):
            raise ValueError(f"parameters {sorted(values)} don't match the model's {sorted(params)}")
        for name, p in params.items():
            if np.shape(values[name]) != p.shape:
                raise ValueError(f"parameter '{name}' has shape {np.shape(values[name])}, expected {p.shape}")
        for name, p in params.items():
            p.value = np.array(values[name], dtype=float)

    def save_parameters(self, path):
        np.savez(path, **self.parameter_values())

    def load_parameters(self, path):
        with np.load(path) as data:
            self.set_parameter_values({k: data[k] for k in data.files})
    # >>>

    def check_dataset(self, dataset, outputs=True):
        missing = [layer.name for layer in self.inputs if layer.name not in dataset.inputs]
        if missing:
            raise KeyError(f"the dataset has no inputs for {missing}")
        if outputs:
            names = {layer.name for layer in self.outputs}
            if set(dataset.outputs) != names:
                raise KeyError(f"dataset outputs {sorted(dataset.outputs)} don't match the output layers {sorted(names)}")
            for layer in self.outputs:
                if dataset.output_sizes[layer.name] != layer.units:
                    raise ValueError(f"output '{layer.name}' has {layer.units} units, the dataset has {dataset.output_sizes[layer.name]}")

    # netlist <<<
    def build_netlist(self, inputs, sample_index=0):
        netlist = Netlist(temperature=self.temperature)
        for layer in self.layers:
            layer.stamp(netlist.block(layer.name), inputs, sample_index)
        return netlist

    def write_netlist(self, dataset, filename, sample=0):
        with open(filename, "w") as f:
            f.write(str(self.build_netlist(dataset.inputs, sample)))
    # >>>

    # training <<<
    def fit(self, *, train_data, epochs, optimizer, loss, algorithm, metrics=None, validation_data=None,
            validation_freq=1, callbacks=(), checkpoint=None, checkpoint_every=None, resume=None, workers=1):
        """
        loss and metrics are used for every output, or given per output in a
        dict. The run is saved to `checkpoint` after every epoch (and every
        `checkpoint_every` batches) and continues exactly from `resume`.
        Samples are simulated in `workers` processes (0: in this process).
        Returns the History.
        """
        self.check_dataset(train_data.dataset)
        if validation_data is not None:
            self.check_dataset(validation_data)
        loss, metrics = per_output(self, loss, metrics)
        training = Training(
            self, train_data=train_data, epochs=epochs, optimizer=optimizer, loss=loss, metrics=metrics,
            algorithm=algorithm, validation_data=validation_data, validation_freq=validation_freq,
            callbacks=callbacks, checkpoint=checkpoint, checkpoint_every=checkpoint_every, workers=workers,
        )
        return training.run(resume)

    def predict(self, dataset, algorithm=None, workers=1):
        """Free phase outputs, unless the algorithm has its own predict_step."""
        self.check_dataset(dataset, outputs=False)
        with Workers(self, workers, algorithm) as w:
            return w.infer(dataset.inputs, len(dataset))

    def evaluate(self, dataset, loss, metrics=None, algorithm=None, workers=1):
        self.check_dataset(dataset)
        loss, metrics = per_output(self, loss, metrics)
        outputs = self.predict(dataset, algorithm, workers)
        return summary(sample_losses(outputs, dataset.outputs, loss), outputs, dataset.outputs, metrics)
    # >>>
