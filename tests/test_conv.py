import numpy as np

from ebana import Dataset, DataLoader, Model
from ebana.algorithms import EquilibriumPropagation
from ebana.constraints import Clip, MaxRelativeChange
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.layers import Amplifier, CurrentOutput, Dense, Differential, Diode, Input, LocallyConnected2D, Reshape, Stack
from ebana.losses import CrossEntropy
from ebana.metrics import CategoricalAccuracy
from ebana.optimizers import Adam
from ebana.schedules import CosineDecay

LR = 5e-9


def images(n, seed):
    """6x6 images with a bright left or right half."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = rng.uniform(-1, -0.6, size=(n, 6, 6))
    for i in range(n):
        x[i][:, 3 * y[i]: 3 * y[i] + 3] = rng.uniform(0.6, 1, size=(6, 3))
    x = x.reshape(n, 36)
    return Dataset({"xp": x, "xn": -x}, {"c1": np.eye(2)[y]})


def build_model():
    init = GlorotUniform(low=1e-7, high=1e-5, seed=111)
    window = [MaxRelativeChange(0.5), Clip(1e-7, 1e-5)]
    diode = IdealDiode(vth=0.15, ron=1.0, roff=1e20)

    xp = Input("xp", 36)
    xn = Input("xn", 36)
    rxp = Reshape("rxp", (6, 6))(xp)
    rxn = Reshape("rxn", (6, 6))(xn)
    image = Stack("j1", axis=-1)([rxp, rxn])
    d1 = LocallyConnected2D("d1", kernel_size=(3, 3), stride=(3, 3), filters=2, initializer=init, constraint=window,
                            lr=CosineDecay(LR, decay_steps=20, alpha=0.1))(image)
    down = Diode("down", direction="down", model=diode, bias_voltage=0.5)(d1)
    up = Diode("up", direction="up", model=diode, bias_voltage=-0.1)(down)
    amp = Amplifier("amp", gain=10, device=BidirectionalAmplifier(vcm=0.0))(up)
    flat = Reshape("r2", (-1,))(amp)
    d_out = Dense("d_out", 4, lr=LR, initializer=init, constraint=window)(flat)
    out = CurrentOutput("c1", readout=Differential())(d_out)
    return Model(outputs=[out])


def test_learns_the_images():
    model = build_model()
    history = model.fit(train_data=DataLoader(images(16, seed=0), 4, shuffle=True, seed=0),
                        validation_data=images(8, seed=1), epochs=8, loss=CrossEntropy(),
                        metrics=[CategoricalAccuracy()], optimizer=Adam(model),
                        algorithm=EquilibriumPropagation(1e-6), workers=4)

    loss = history.epoch_values("loss/c1")
    accuracy = history.epoch_values("val_categorical_accuracy/c1")
    assert np.all(np.diff(loss) < 0)
    assert accuracy[0] < 1 and accuracy[-1] == 1
