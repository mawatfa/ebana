"""
Callbacks run during Model.fit. In a batch, the samples are simulated, the
optimizer steps, the history is updated and then on_batch_end runs, so a
callback can also reprogram the parameters there (quantization, write
noise...). Callbacks with state implement state_dict and load_state_dict so
that it is saved in checkpoints.
"""

import logging

import numpy as np

log = logging.getLogger("ebana")


class Callback:
    probes = {}     # name -> probe, measured after every phase when recording

    def recording(self, training):
        return bool(self.probes)

    def on_train_begin(self, training):
        pass

    def on_batch_end(self, training, logs):
        pass

    def on_epoch_end(self, training, logs):
        pass

    def on_train_end(self, training):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


# history <<<
def _columns(records):
    keys = dict.fromkeys(k for r in records for k in r)
    return {k: np.array([r.get(k, np.nan) for r in records], dtype=float) for k in keys}


class History:
    """
    One dict per batch and one per epoch, with keys like "loss", "loss/out",
    "categorical_accuracy/out" and, for validation, "val_loss/out".
    """

    def __init__(self):
        self.batches = []
        self.epochs = []

    def batch_values(self, key):
        return _columns(self.batches).get(key, np.full(len(self.batches), np.nan))

    def epoch_values(self, key):
        """NaN for the epochs without that key (e.g. no validation)."""
        return _columns(self.epochs).get(key, np.full(len(self.epochs), np.nan))

    def save(self, path):
        arrays = {f"batch/{k}": v for k, v in _columns(self.batches).items()}
        arrays |= {f"epoch/{k}": v for k, v in _columns(self.epochs).items()}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        history = cls()
        with np.load(path) as data:
            for group, records in (("batch", history.batches), ("epoch", history.epochs)):
                columns = {k.split("/", 1)[1]: data[k] for k in data.files if k.startswith(group + "/")}
                n = len(next(iter(columns.values()))) if columns else 0
                records.extend({k: float(v[i]) for k, v in columns.items() if not np.isnan(v[i])} for i in range(n))
        return history

    def state_dict(self):
        return {"batches": [dict(r) for r in self.batches], "epochs": [dict(r) for r in self.epochs]}

    def load_state_dict(self, state):
        self.batches = [dict(r) for r in state["batches"]]
        self.epochs = [dict(r) for r in state["epochs"]]
# >>>


# recorder <<<
class Recorder(Callback):
    """
    Records every sample of one batch in `every`: the probes after every phase
    ("<probe>/<phase>"), and optionally the outputs, losses and forces
    ("output/<output>", ...). get(key) stacks them into (records, batch, ...).
    """

    def __init__(self, probes=None, every=1, outputs=False, losses=False, forces=False):
        self.probes = dict(probes or {})
        self.every = every
        self.keep = {"output": outputs, "loss": losses, "force": forces}
        self.records = []

    def recording(self, training):
        return training.step % self.every == 0

    def on_batch_end(self, training, logs):
        if not self.recording(training):
            return
        s = training.samples
        record = {"step": training.step, "epoch": training.epoch, "batch": training.batch, "index": s["index"]}
        for kind, wanted in self.keep.items():
            if wanted:
                record |= {f"{kind}/{name}": v for name, v in s[kind].items()}
        for name in self.probes:
            record |= {f"{name}/{phase}": v for phase, v in s["probes"][name].items()}
        self.records.append(record)

    def keys(self):
        return list(dict.fromkeys(k for r in self.records for k in r))

    def get(self, key):
        return np.stack([r[key] for r in self.records])

    def save(self, path):
        np.savez(path, **{k: self.get(k) for k in self.keys()})

    def state_dict(self):
        return {"records": list(self.records)}

    def load_state_dict(self, state):
        self.records = list(state["records"])
# >>>


# early stopping <<<
class EarlyStopping(Callback):
    """Stops when `monitor` hasn't improved for `patience` epochs, optionally keeping the best parameters."""

    def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, mode="min", restore_best=False):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.sign = 1 if mode == "min" else -1
        self.restore_best = restore_best
        self.best, self.wait, self.best_values = None, 0, None

    def on_epoch_end(self, training, logs):
        if self.monitor not in logs:
            return
        value = logs[self.monitor]
        if self.best is None or self.sign * (value - self.best) < -self.min_delta:
            self.best, self.wait = value, 0
            if self.restore_best:
                self.best_values = training.model.parameter_values()
            return
        self.wait += 1
        if self.wait >= self.patience:
            log.info(f"early stopping: no improvement of {self.monitor} in {self.wait} epochs")
            training.stop = True

    def on_train_end(self, training):
        if self.restore_best and self.best_values is not None:
            training.model.set_parameter_values(self.best_values)

    def state_dict(self):
        return {"best": self.best, "wait": self.wait, "best_values": self.best_values}

    def load_state_dict(self, state):
        self.best, self.wait, self.best_values = state["best"], state["wait"], state["best_values"]
# >>>
