import logging
import os
import pickle

import numpy as np

from . import evaluator
from .callbacks import History
from .execution import Workers

log = logging.getLogger("ebana")

CHECKPOINT_VERSION = 1


def _format(logs):
    return "  ".join(f"{k} {v:.6g}" for k, v in logs.items())


class Training:
    """
    One call of Model.fit. Callbacks receive it and can read the model,
    optimizer and history, the position (epoch, batch, step), the results of
    the last batch in `samples`, and set `stop` to end training.
    """

    def __init__(self, model, *, train_data, epochs, optimizer, loss, metrics, algorithm,
                 validation_data, validation_freq, callbacks, checkpoint, checkpoint_every, workers):
        self.model, self.train_data, self.epochs = model, train_data, epochs
        self.optimizer, self.loss, self.metrics, self.algorithm = optimizer, loss, metrics, algorithm
        self.validation_data, self.validation_freq = validation_data, validation_freq
        self.callbacks = list(callbacks)
        self.checkpoint, self.checkpoint_every = checkpoint, checkpoint_every
        self.n_workers = workers
        if checkpoint is not None and train_data.shuffle and train_data.seed is None:
            raise ValueError("checkpoints need a DataLoader seed, otherwise a resumed run sees other batches")
        self.history = History()
        self.epoch = self.batch = self.step = 0
        self.stop = False
        self.samples = None
        self.probes = {name: probe for c in self.callbacks for name, probe in c.probes.items()}

    # running <<<
    def run(self, resume=None):
        if resume is not None:
            self.load(resume)
        if self.epoch >= self.epochs and not self.stop:
            raise ValueError(f"the checkpoint is at epoch {self.epoch}, nothing to do with epochs={self.epochs}")
        with Workers(self.model, self.n_workers, self.algorithm, self.loss, self.probes) as self.workers:
            for callback in self.callbacks:
                callback.on_train_begin(self)
            while self.epoch < self.epochs and not self.stop:
                batches = list(self.train_data.batches(self.epoch))
                while self.batch < len(batches) and not self.stop:
                    self._run_batch(*batches[self.batch])
                if self.stop:
                    break
                self._end_epoch()
            for callback in self.callbacks:
                callback.on_train_end(self)
        return self.history

    def _run_batch(self, indices, batch):
        names = [n for c in self.callbacks if c.probes and c.recording(self) for n in c.probes]
        grads, samples = self.workers.train(batch.inputs, batch.outputs, names)
        for p in self.model.trainable_parameters():
            p.grad = grads[p.full_name] / len(indices)
        self.samples = self._collect(indices, samples)
        self.optimizer.step()

        logs = {"epoch": self.epoch, "batch": self.batch, "step": self.step}
        logs |= evaluator.summary(self.samples["loss"], self.samples["output"], batch.outputs, self.metrics)
        self.history.batches.append(logs)
        log.debug(f"epoch {self.epoch + 1} batch {self.batch + 1}: {_format(logs)}")
        for callback in self.callbacks:
            callback.on_batch_end(self, logs)

        self.batch += 1
        self.step += 1
        if self.checkpoint is not None and self.checkpoint_every and self.step % self.checkpoint_every == 0:
            self.save(self.checkpoint)

    def _collect(self, indices, samples):
        def stack(kind):
            return {name: np.stack([np.asarray(s[kind][name]) for s in samples]) for name in samples[0][kind]}
        return {
            "index": np.asarray(indices),
            "output": stack("output"),
            "loss": stack("loss"),
            "force": stack("force"),
            "probes": {name: {phase: np.stack([s["probes"][name][phase] for s in samples])
                              for phase in samples[0]["probes"][name]}
                       for name in samples[0]["probes"]},
        }

    def _end_epoch(self):
        batches = [r for r in self.history.batches if r["epoch"] == self.epoch]
        keys = [k for k in batches[0] if k not in ("epoch", "batch", "step")]
        logs = {"epoch": self.epoch} | {k: float(np.mean([r[k] for r in batches])) for k in keys}
        if self.validation_data is not None and (self.epoch + 1) % self.validation_freq == 0:
            logs |= {f"val_{k}": v for k, v in self.validate().items()}
        self.history.epochs.append(logs)
        log.info(f"epoch {self.epoch + 1}/{self.epochs}: {_format({k: v for k, v in logs.items() if k != 'epoch'})}")
        for callback in self.callbacks:
            callback.on_epoch_end(self, logs)

        self.epoch += 1
        self.batch = 0
        if self.checkpoint is not None:
            self.save(self.checkpoint)

    def validate(self):
        data = self.validation_data
        outputs = self.workers.infer(data.inputs, len(data))
        return evaluator.summary(evaluator.sample_losses(outputs, data.outputs, self.loss),
                                 outputs, data.outputs, self.metrics)
    # >>>

    # checkpoints <<<
    def _data_settings(self):
        loader = self.train_data
        return {"samples": len(loader.dataset), "batch_size": loader.batch_size,
                "shuffle": loader.shuffle, "seed": loader.seed}

    def _callback_states(self):
        return {f"{i}:{type(c).__name__}": c.state_dict() for i, c in enumerate(self.callbacks)}

    def save(self, path):
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "position": {"epoch": self.epoch, "batch": self.batch, "step": self.step, "stop": self.stop},
            "data": self._data_settings(),
            "parameters": self.model.parameter_values(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history.state_dict(),
            "callbacks": self._callback_states(),
        }
        # write to a temporary file first so a crash can't corrupt the last checkpoint
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path} is not a training checkpoint")
        if checkpoint["data"] != self._data_settings():
            raise ValueError(f"{path} was saved with other training data: {checkpoint['data']}")
        if set(checkpoint["callbacks"]) != set(self._callback_states()):
            raise ValueError(f"{path} was saved with other callbacks: {sorted(checkpoint['callbacks'])}")

        before = self.model.parameter_values()
        try:
            self.model.set_parameter_values(checkpoint["parameters"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            self.model.set_parameter_values(before)
            raise
        self.history.load_state_dict(checkpoint["history"])
        for key, callback in zip(self._callback_states(), self.callbacks):
            callback.load_state_dict(checkpoint["callbacks"][key])
        position = checkpoint["position"]
        self.epoch, self.batch, self.step, self.stop = (position[k] for k in ("epoch", "batch", "step", "stop"))
        log.info(f"resuming from {path} at epoch {self.epoch + 1}, batch {self.batch + 1}")
    # >>>
