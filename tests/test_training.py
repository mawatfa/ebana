from itertools import product

import numpy as np
import pytest

from ebana import Dataset, DataLoader, Model
from ebana.algorithms import EquilibriumPropagation
from ebana.callbacks import Callback, EarlyStopping, Recorder
from ebana.constraints import Clip, MaxRelativeChange
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.layers import Amplifier, Bias, Concatenate, CurrentOutput, Dense, Differential, Diode, Input
from ebana.losses import MSE
from ebana.metrics import BinaryAccuracy
from ebana.optimizers import SGD, Adam
from ebana.probes import Measurement, Voltages

BETA = 2.73e-11
GATES = {"or": [0, 1, 1, 1], "and": [0, 0, 0, 1], "xor": [0, 1, 1, 0]}


def build_model():
    init = GlorotUniform(low=1e-7, high=1e-5, seed=111)
    window = [MaxRelativeChange(0.5), Clip(1e-7, 1e-5)]
    diode = IdealDiode(vth=0.3, ron=1.0, roff=1e20)
    lr = {"d1": 1.036e-9, "or": 9.08e-11, "and": 4.64e-11, "xor": 4.76e-11}

    j1 = Concatenate("j1")([Input("xp", 2), Input("xn", 2), Bias("b1", [1.0])])
    h = Dense("d1", 10, lr=lr["d1"], initializer=init, constraint=window)(j1)
    h = Diode("down", direction="down", model=diode)(h)
    h = Diode("up", direction="up", model=diode)(h)
    h = Amplifier("amp", gain=4, device=BidirectionalAmplifier(shift=0))(h)
    j2 = Concatenate("j2")([h, Bias("b2", [1.0])])
    outputs = [CurrentOutput(gate, readout=Differential())(Dense(f"d_{gate}", 2, lr=lr[gate], initializer=init,
                                                               constraint=window)(j2))
               for gate in GATES]
    return Model(outputs=outputs)


def gates():
    x = np.array(list(product([-2, 2], repeat=2)))
    return Dataset({"xp": x, "xn": -x}, {gate: np.array(y).reshape(-1, 1) for gate, y in GATES.items()})


def fit(model, epochs=10, batch_size=4, optimizer=SGD, **options):
    return model.fit(train_data=DataLoader(gates(), batch_size, shuffle=True, seed=3), validation_data=gates(),
                     epochs=epochs, loss=MSE(), metrics=[BinaryAccuracy(0.5)], optimizer=optimizer(model),
                     algorithm=EquilibriumPropagation(BETA), workers=2, **options)


def test_learns_the_gates():
    history = fit(build_model())
    for gate in GATES:
        loss = history.epoch_values(f"loss/{gate}")
        assert loss[-1] < loss[0] / 5
        assert history.epoch_values(f"val_binary_accuracy/{gate}")[-1] == 1


def test_workers_do_not_change_the_outputs():
    model = build_model()
    one, three = model.predict(gates(), workers=0), model.predict(gates(), workers=3)
    for gate in GATES:
        np.testing.assert_allclose(one[gate], three[gate], rtol=1e-12)


class Noise(Callback):
    """Programs d1 with a random error after every update."""

    def __init__(self):
        self.rng = np.random.default_rng(5)

    def on_batch_end(self, training, logs):
        w = training.model.parameter("d1/weight")
        w.value = w.value * (1 + 0.01 * self.rng.standard_normal(w.shape))

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


class Crash(Callback):
    def __init__(self, step=None):
        self.step = step

    def on_batch_end(self, training, logs):
        if training.step == self.step:
            raise RuntimeError("crash")


def run(model, crash=None, **options):
    callbacks = [Crash(crash), Noise(), Recorder({"w": Measurement("d1/weight"), "h": Voltages("amp")}, every=3),
                 EarlyStopping(patience=100, restore_best=True)]
    history = fit(model, epochs=5, batch_size=2, optimizer=Adam, callbacks=callbacks, **options)
    return history, callbacks


def test_resume_after_a_crash(tmp_path):
    model = build_model()
    history, callbacks = run(model)

    with pytest.raises(RuntimeError):
        run(build_model(), crash=7, checkpoint=tmp_path / "ckpt", checkpoint_every=1)
    resumed = build_model()
    resumed_history, resumed_callbacks = run(resumed, resume=tmp_path / "ckpt")

    assert resumed_history.batches == history.batches
    assert resumed_history.epochs == history.epochs
    for name, value in model.parameter_values().items():
        np.testing.assert_array_equal(resumed.parameter(name).value, value)
    recorder, resumed_recorder = callbacks[2], resumed_callbacks[2]
    assert recorder.get("step").tolist() == [0, 3, 6, 9]
    for key in recorder.keys():
        np.testing.assert_array_equal(resumed_recorder.get(key), recorder.get(key))


def test_saved_files(tmp_path):
    model = build_model()
    history = fit(model, epochs=2)
    history.save(tmp_path / "history.npz")
    model.save_parameters(tmp_path / "parameters.npz")

    other = build_model()
    other.load_parameters(tmp_path / "parameters.npz")
    assert other.evaluate(gates(), MSE()) == model.evaluate(gates(), MSE())
    with np.load(tmp_path / "history.npz") as saved:
        np.testing.assert_array_equal(saved["epoch/loss"], history.epoch_values("loss"))
