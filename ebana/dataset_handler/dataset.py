import numpy as np


class Dataset:
    """
    Inputs and outputs by layer name. The first axis is the sample; outputs
    are (samples, units) and can be left out for prediction.
    """

    def __init__(self, inputs, outputs=None):
        self.inputs = {name: np.asarray(v) for name, v in inputs.items()}
        self.outputs = {name: np.asarray(v) for name, v in (outputs or {}).items()}
        for name, v in self.outputs.items():
            if v.ndim != 2:
                raise ValueError(f"output '{name}' must be 2-D (samples, units), got shape {v.shape}")
        sizes = {len(v) for v in [*self.inputs.values(), *self.outputs.values()]}
        if len(sizes) != 1:
            raise ValueError("all inputs and outputs must have the same number of samples")
        self._size = sizes.pop()

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            index = [index]
        return Dataset(
            {name: v[index] for name, v in self.inputs.items()},
            {name: v[index] for name, v in self.outputs.items()},
        )

    @property
    def output_sizes(self):
        return {name: v.shape[1] for name, v in self.outputs.items()}

    def split(self, first, shuffle=True, seed=None):
        """Two datasets, the first with `first` samples (or that fraction of them)."""
        n = round(first * len(self)) if isinstance(first, float) else first
        order = np.random.default_rng(seed).permutation(len(self)) if shuffle else np.arange(len(self))
        return self[order[:n]], self[order[n:]]
