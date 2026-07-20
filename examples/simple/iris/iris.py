"""
Iris: three classes from four flower measurements.

    xp, xn, 2 bias rows -> d1 (10) -> diode clamps -> amp -> 2 bias rows -> d2 -> "out" (3)

The circuit works about a common-mode voltage VCM: inputs are VCM +- x_scale,
the diode clamps and the amplifier are referenced to VCM, and each class
score is the difference of two output nets. The loss is cross-entropy on the
scores, or MSE towards target voltages.

    python iris.py             train with ARGS below (see "mode" for the other runs)
    python iris_test.py        train a list of configurations, one log file each
    python iris_optimizer.py   tune beta and the learning rates with optuna
"""

# imports <<<
import logging
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

from ebana import Dataset, DataLoader, Model, gradient_check
from ebana import optimizers
from ebana.algorithms import EquilibriumPropagation, SymmetricEquilibriumPropagation
from ebana.callbacks import EarlyStopping, Recorder
from ebana.constraints import Clip, MaxRelativeChange
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.layers import Amplifier, Bias, Concatenate, CurrentOutput, Dense, Differential, Diode, Input
from ebana.losses import MSE, CrossEntropy
from ebana.metrics import CategoricalAccuracy
from ebana.probes import Measurement, Voltages
from ebana.schedules import CosineDecay
# >>>

log = logging.getLogger("iris")

# settings <<<
ARGS = {
    # run
    "mode": "train",            # "train", "evaluate" (saved parameters), "netlist" or "gradcheck"
    "name": None,               # name of the run's files in runs/ (None: built from the settings)
    "workers": 8,               # processes simulating samples in parallel
    "seed": 111,
    "save": True,               # checkpoint during training, save history and parameters
    "resume": False,            # continue from the run's checkpoint, if there is one
    "verbose": True,
    # data
    "train_samples": 105,       # the other 45 flowers are the validation set
    "vcm": 0.6,                 # common-mode voltage of the whole circuit
    "x_scale": 0.6,             # features become VCM +- x_scale
    "y_scale": 1.2,             # MSE targets: y_scale for the right class, 0 for the others
    # circuit
    "hidden_units": 10,
    "g_range": (1e-7, 1e-5),    # conductance window of the crossbars (S)
    "diode_vth": 0.15,          # the clamps conduct beyond VCM +- 0.15 V
    "gain": 4,
    # training
    "epochs": 40,
    "batch_size": 35,
    "loss": "crossentropy",     # "crossentropy" or "mse"
    "algorithm": "ep",          # "ep" or "symmetric_ep"
    "beta": 2e-9,               # nudge strength (S)
    "optimizer": "sgd",         # "sgd", "momentum" or "adam"
    "lr_hidden": 1e-8,
    "lr_out": 1e-9,
    "lr_schedule": "cosine",    # "constant" or "cosine" (decays to 10% over the run)
    "update_rule": "device",    # "device" (its own gradient), "power" or "product" (see below)
    "update_exponent": 2.8,     # p of the "power" rule
    "clip_norm": None,          # clip each gradient to this L2 norm
    "patience": None,           # stop after this many epochs without a better validation loss
    "record_every": None,       # record hidden voltages and weight drops every n batches
}
# >>>

# data <<<
def datasets(args):
    """Train and validation sets. Features are scaled to [-1, 1] with the training set's range."""
    iris = load_iris()
    order = np.random.default_rng(args["seed"]).permutation(len(iris.target))
    x, labels = iris.data[order], iris.target[order]
    n = args["train_samples"]
    low, high = x[:n].min(axis=0), x[:n].max(axis=0)
    x = args["x_scale"] * np.clip(2 * (x - low) / (high - low) - 1, -1, 1)

    y = np.eye(3)[labels]
    if args["loss"] == "mse":
        y = args["y_scale"] * y

    vcm = args["vcm"]
    data = Dataset(inputs={"xp": vcm + x, "xn": vcm - x}, outputs={"out": y})
    return data[:n], data[n:]
# >>>

# options <<<
class PowerRule:
    """For a device that responds to its voltage like |V|^p. p = 2 is the normal EP rule."""

    def __init__(self, p):
        self.p = p

    def __call__(self, a, b):
        return (np.abs(b)**self.p - np.abs(a)**self.p) / self.p


class ProductRule:
    """The EP rule linearized around the first state: one multiplication per device."""

    def __call__(self, a, b):
        return a * (b - a)


def update_rule(args):
    return {"device": None, "power": PowerRule(args["update_exponent"]), "product": ProductRule()}[args["update_rule"]]


def learning_rate(args, lr, steps):
    if args["lr_schedule"] == "cosine":
        return CosineDecay(lr, decay_steps=steps, alpha=0.1)
    return lr


def algorithm(args):
    kind = {"ep": EquilibriumPropagation, "symmetric_ep": SymmetricEquilibriumPropagation}[args["algorithm"]]
    return kind(args["beta"])


def optimizer(args, model):
    transforms = [optimizers.ClipByNorm(args["clip_norm"])] if args["clip_norm"] else []
    if args["optimizer"] == "adam":
        return optimizers.Adam(model, transforms=transforms)
    if args["optimizer"] == "momentum":
        return optimizers.SGDMomentum(model, momentum=0.9, transforms=transforms)
    return optimizers.SGD(model, transforms=transforms)


def callbacks(args):
    chosen = []
    if args["patience"]:
        chosen.append(EarlyStopping("val_loss", patience=args["patience"], restore_best=True))
    if args["record_every"]:
        probes = {"hidden": Voltages("amp"), "drops": Measurement("d1/weight")}
        chosen.append(Recorder(probes, every=args["record_every"], outputs=True, forces=True))
    return chosen
# >>>

# model <<<
def build_model(args, steps):
    g_min, g_max = args["g_range"]
    init = GlorotUniform(low=g_min, high=g_max, seed=args["seed"])
    window = [MaxRelativeChange(0.5), Clip(g_min, g_max)]
    diode = IdealDiode(vth=args["diode_vth"], ron=1.0, roff=1e20)
    rule = update_rule(args)
    vcm, half = args["vcm"], args["x_scale"] / 2
    bias_rows = [vcm + half, vcm - half]

    def crossbar(name, units, lr):
        return Dense(name, units, initializer=init, constraint=window, grad_func=rule,
                     lr=learning_rate(args, lr, steps))

    # hidden layer: inputs and two bias rows through a crossbar, clamped about VCM, amplified about VCM
    xp = Input("xp", 4)
    xn = Input("xn", 4)
    bias1 = Bias("bias1", bias_rows)
    rows1 = Concatenate("rows1")([xp, xn, bias1])
    d1 = crossbar("d1", args["hidden_units"], args["lr_hidden"])(rows1)
    clamp_hi = Diode("clamp_hi", "down", model=diode, bias_voltage=vcm)(d1)
    clamp_lo = Diode("clamp_lo", "up", model=diode, bias_voltage=vcm)(clamp_hi)
    amp = Amplifier("amp", gain=args["gain"], device=BidirectionalAmplifier(vcm=vcm))(clamp_lo)

    # output layer: two nets per class, read as their difference
    bias2 = Bias("bias2", bias_rows)
    rows2 = Concatenate("rows2")([amp, bias2])
    d2 = crossbar("d2", 2 * 3, args["lr_out"])(rows2)
    out = CurrentOutput("out", readout=Differential())(d2)
    return Model(outputs=[out])
# >>>

# running <<<
def run_file(args, suffix):
    name = args["name"] or "_".join([
        "iris", args["loss"], args["algorithm"], args["optimizer"], args["update_rule"],
        f"beta{args['beta']:.2e}", f"lr{args['lr_hidden']:.2e}_{args['lr_out']:.2e}",
    ])
    Path("runs").mkdir(exist_ok=True)
    return Path("runs") / f"{name}{suffix}"


def check_gradients(model, data, loss, args):
    """Should be 1, or 1/gain^2 before the amplifier."""
    results = gradient_check(model, data, loss, algorithm(args), elements=5, seed=args["seed"])
    for name, r in results.items():
        e, n = r["estimate"], r["numerical"]
        ok = np.abs(n) > 1e-9 * np.abs(n).max() if np.any(n) else np.zeros(len(n), bool)
        log.info(f"{name}: estimate / finite difference = {np.median(e[ok] / n[ok]) if ok.any() else np.nan:.4g}")
    return results


def train(config=None):
    """Runs the example with ARGS updated by config."""
    args = {**ARGS, **(config or {})}
    logging.getLogger("ebana").setLevel(logging.INFO if args["verbose"] else logging.WARNING)

    train_data, val_data = datasets(args)
    loader = DataLoader(train_data, args["batch_size"], shuffle=True, seed=args["seed"])
    model = build_model(args, steps=args["epochs"] * len(loader))
    loss = CrossEntropy() if args["loss"] == "crossentropy" else MSE()
    metrics = [CategoricalAccuracy()]

    if args["mode"] == "netlist":
        model.write_netlist(train_data, run_file(args, ".sp"))
        return run_file(args, ".sp")
    if args["mode"] == "gradcheck":
        return check_gradients(model, train_data, loss, args)
    if args["mode"] == "evaluate":
        model.load_parameters(run_file(args, "_parameters.npz"))
        results = model.evaluate(val_data, loss, metrics, workers=args["workers"])
        log.info("  ".join(f"{k} {v:.4g}" for k, v in results.items()))
        return results

    checkpoint = run_file(args, ".ckpt")
    run_callbacks = callbacks(args)
    history = model.fit(
        train_data=loader,
        validation_data=val_data,
        epochs=args["epochs"],
        optimizer=optimizer(args, model),
        algorithm=algorithm(args),
        loss=loss,
        metrics=metrics,
        callbacks=run_callbacks,
        checkpoint=checkpoint if args["save"] else None,
        resume=checkpoint if args["resume"] and checkpoint.exists() else None,
        workers=args["workers"],
    )
    if args["save"]:
        history.save(run_file(args, "_history.npz"))
        model.save_parameters(run_file(args, "_parameters.npz"))
        for callback in run_callbacks:
            if isinstance(callback, Recorder):
                callback.save(run_file(args, "_recorded.npz"))
    return history
# >>>


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train()
