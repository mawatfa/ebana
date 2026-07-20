"""
MNIST with a locally connected (convolution-like) hidden layer.

    xp, xn (28 x 28) -> stack (28 x 28 x 2) -> d1: 5 x 5 patches, stride 3, 2 filters
                     -> diode clamps -> amp -> flatten -> d2 -> "out" (10)

Every output pixel of d1 has its own crossbar over its patch of both input
polarities (the weights are not shared, as in a physical array). The data
are the .npy files in ARGS["data_dir"] (x_train.npy: (samples, 784) pixels,
y_train.npy: one-hot labels). Validation uses the training samples
themselves: the example shows how far the circuit can fit them.

    python mnist.py             train with ARGS below (see "mode" for the other runs)
    python mnist_test.py        train a list of configurations, one log file each
    python mnist_optimizer.py   tune beta and the learning rates with optuna
"""

# imports <<<
import logging
from pathlib import Path

import numpy as np

from ebana import Dataset, DataLoader, Model, gradient_check
from ebana import optimizers
from ebana.algorithms import EquilibriumPropagation, SymmetricEquilibriumPropagation
from ebana.callbacks import EarlyStopping, Recorder
from ebana.constraints import Clip, MaxRelativeChange
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.layers import (Amplifier, CurrentOutput, Dense, Differential, Diode, Input, LocallyConnected2D,
                          Reshape, Stack)
from ebana.losses import MSE, CrossEntropy
from ebana.metrics import CategoricalAccuracy
from ebana.probes import Measurement, Voltages
from ebana.schedules import CosineDecay
# >>>

log = logging.getLogger("mnist_conv")

# settings <<<
ARGS = {
    # run
    "mode": "train",            # "train", "evaluate" (saved parameters), "netlist" or "gradcheck"
    "name": None,               # name of the run's files in runs/ (None: built from the settings)
    "workers": 40,              # processes simulating samples in parallel
    "seed": 111,
    "save": True,               # checkpoint during training, save history and parameters
    "resume": False,            # continue from the run's checkpoint, if there is one
    "verbose": True,
    # data
    "data_dir": "/home/medwatt/coding/ebana_datasets/mnist_full",
    "samples": 500,             # training samples, also used for validation
    # circuit
    "kernel_size": 5,
    "stride": 3,
    "filters": 2,
    "padding": "valid",         # "valid" or "same" (the border is padded with ground)
    "g_range": (1e-7, 1e-5),    # conductance window of the crossbars (S)
    "diode_vth": 0.15,
    "clamp_hi": 0.5,            # bias of the upper clamp: conducts above 0.65 V
    "clamp_lo": 0.1,            # bias of the lower clamp: conducts below -0.05 V
    "gain": 10,
    # training
    "epochs": 5,
    "batch_size": 40,
    "loss": "crossentropy",     # "crossentropy" or "mse"
    "algorithm": "ep",          # "ep" or "symmetric_ep"
    "beta": 1e-6,               # nudge strength (S)
    "optimizer": "adam",        # "sgd", "momentum" or "adam"
    "lr_hidden": 1e-7,
    "lr_out": 1e-7,
    "lr_schedule": "constant",  # "constant" or "cosine" (decays to 10% over the run)
    "update_rule": "device",    # "device" (its own gradient), "power" or "product" (see below)
    "update_exponent": 2.8,     # p of the "power" rule
    "clip_norm": None,          # clip each gradient to this L2 norm
    "patience": None,           # stop after this many epochs without a better validation loss
    "record_every": None,       # record hidden voltages and output weight drops every n batches
}
# >>>

# data <<<
PIXEL_MEAN, PIXEL_STD = 33.318421449829934, 78.56748998339798     # over the full training set

def datasets(args):
    """Train and validation sets: here the same standardized training samples."""
    directory, n = Path(args["data_dir"]), args["samples"]
    x = (np.load(directory / "x_train.npy")[:n] - PIXEL_MEAN) / PIXEL_STD
    y = np.load(directory / "y_train.npy")[:n].astype(float)
    data = Dataset(inputs={"xp": x, "xn": -x}, outputs={"out": y})
    return data, data
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
        probes = {"hidden": Voltages("amp"), "drops": Measurement("d2/weight")}
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
    k, s = args["kernel_size"], args["stride"]

    # both polarities of the image, stacked as channels: (28, 28, 2)
    xp = Input("xp", 784)
    xn = Input("xn", 784)
    xp_2d = Reshape("xp_2d", (28, 28))(xp)
    xn_2d = Reshape("xn_2d", (28, 28))(xn)
    image = Stack("image", axis=-1)([xp_2d, xn_2d])

    # hidden layer: one crossbar per output pixel over its patch, clamped, amplified
    d1 = LocallyConnected2D("d1", kernel_size=(k, k), stride=(s, s), filters=args["filters"],
                            padding=args["padding"], initializer=init, constraint=window, grad_func=rule,
                            lr=learning_rate(args, args["lr_hidden"], steps))(image)
    clamp_hi = Diode("clamp_hi", "down", model=diode, bias_voltage=args["clamp_hi"])(d1)
    clamp_lo = Diode("clamp_lo", "up", model=diode, bias_voltage=args["clamp_lo"])(clamp_hi)
    amp = Amplifier("amp", gain=args["gain"], device=BidirectionalAmplifier(vcm=0.0))(clamp_lo)

    # output layer: two nets per class, read as their difference
    flat = Reshape("flat", (-1,))(amp)
    d2 = Dense("d2", 2 * 10, initializer=init, constraint=window, grad_func=rule,
               lr=learning_rate(args, args["lr_out"], steps))(flat)
    out = CurrentOutput("out", readout=Differential())(d2)
    return Model(outputs=[out])
# >>>

# running <<<
def run_file(args, suffix):
    name = args["name"] or "_".join([
        "mnist_conv", args["loss"], args["algorithm"], args["optimizer"], args["update_rule"],
        f"k{args['kernel_size']}s{args['stride']}f{args['filters']}",
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
