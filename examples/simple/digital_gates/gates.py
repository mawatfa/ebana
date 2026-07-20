"""
Digital gates: one circuit learns OR, AND and XOR of two bits at once.

    xp, xn, 1 V -> d1 (10) -> diode clamps -> amp -> 1 V -> d_or  -> "or"
                                                         -> d_and -> "and"
                                                         -> d_xor -> "xor"

Conductances are positive, so every input is applied with both polarities
(xp = x, xn = -x), and every output is the difference of two nets. Each
output has its own crossbar and learning rate.

    python gates.py             train with ARGS below (see "mode" for the other runs)
    python gates_test.py        train a list of configurations, one log file each
    python gates_optimizer.py   tune beta and the learning rates with optuna
"""

# imports <<<
import logging
from itertools import product
from pathlib import Path

import numpy as np

from ebana import Dataset, DataLoader, Model, gradient_check
from ebana import optimizers
from ebana.algorithms import EquilibriumPropagation, SymmetricEquilibriumPropagation
from ebana.callbacks import EarlyStopping, Recorder
from ebana.constraints import Clip, MaxRelativeChange
from ebana.devices import BidirectionalAmplifier, IdealDiode
from ebana.initializers import GlorotUniform
from ebana.layers import Amplifier, Bias, Concatenate, CurrentOutput, Dense, Differential, Diode, Input
from ebana.losses import MSE
from ebana.metrics import BinaryAccuracy
from ebana.probes import Measurement, Voltages
from ebana.schedules import CosineDecay
# >>>

log = logging.getLogger("gates")

GATES = {"or": [0, 1, 1, 1], "and": [0, 0, 0, 1], "xor": [0, 1, 1, 0]}

# settings <<<
ARGS = {
    # run
    "mode": "train",            # "train", "evaluate" (saved parameters), "netlist" or "gradcheck"
    "name": None,               # name of the run's files in runs/ (None: built from the settings)
    "workers": 4,               # processes simulating samples in parallel
    "seed": 111,
    "save": True,               # checkpoint during training, save history and parameters
    "resume": False,            # continue from the run's checkpoint, if there is one
    "verbose": True,
    # data
    "bit_voltage": 2.0,         # bits 0 and 1 are -2 V and +2 V
    # circuit
    "hidden_units": 10,
    "g_range": (1e-7, 1e-5),    # conductance window of the crossbars (S)
    "diode_vth": 0.3,           # the clamps conduct beyond +-0.3 V
    "gain": 4,
    # training
    "epochs": 10,
    "batch_size": 4,
    "algorithm": "ep",          # "ep" or "symmetric_ep"
    "beta": 2.73e-11,           # nudge strength (S)
    "optimizer": "sgd",         # "sgd", "momentum" or "adam"
    "lr_hidden": 1.036e-9,
    "lr_or": 9.08e-11,
    "lr_and": 4.64e-11,
    "lr_xor": 4.76e-11,
    "lr_schedule": "constant",  # "constant" or "cosine" (decays to 10% over the run)
    "update_rule": "device",    # "device" (its own gradient), "power" or "product" (see below)
    "update_exponent": 2.8,     # p of the "power" rule
    "clip_norm": None,          # clip each gradient to this L2 norm
    "patience": None,           # stop after this many epochs without a better validation loss
    "record_every": None,       # record hidden voltages and weight drops every n batches
}
# >>>

# data <<<
def datasets(args):
    """Train and validation data: the four input pairs, with each gate's truth table as its output."""
    x = args["bit_voltage"] * np.array(list(product([-1, 1], repeat=2)))
    data = Dataset(
        inputs={"xp": x, "xn": -x},
        outputs={gate: np.array(y).reshape(-1, 1) for gate, y in GATES.items()},
    )
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

    def crossbar(name, units, lr):
        return Dense(name, units, initializer=init, constraint=window, grad_func=rule,
                     lr=learning_rate(args, lr, steps))

    # hidden layer: the inputs and a 1 V bias row through a crossbar, clamped by
    # two diodes (to 0 V, conducting beyond +-vth), then amplified
    xp = Input("xp", 2)
    xn = Input("xn", 2)
    bias1 = Bias("bias1", [1.0])
    rows1 = Concatenate("rows1")([xp, xn, bias1])
    d1 = crossbar("d1", args["hidden_units"], args["lr_hidden"])(rows1)
    clamp_hi = Diode("clamp_hi", "down", model=diode)(d1)
    clamp_lo = Diode("clamp_lo", "up", model=diode)(clamp_hi)
    amp = Amplifier("amp", gain=args["gain"], device=BidirectionalAmplifier(vcm=0.0))(clamp_lo)

    # one output crossbar per gate, all fed by the hidden layer and a bias row
    bias2 = Bias("bias2", [1.0])
    rows2 = Concatenate("rows2")([amp, bias2])
    outputs = []
    for gate in GATES:
        d_gate = crossbar(f"d_{gate}", 2, args[f"lr_{gate}"])(rows2)
        outputs.append(CurrentOutput(gate, readout=Differential())(d_gate))
    return Model(outputs=outputs)
# >>>

# running <<<
def run_file(args, suffix):
    name = args["name"] or "_".join([
        "gates", args["algorithm"], args["optimizer"], args["update_rule"],
        f"beta{args['beta']:.2e}", f"lr{args['lr_hidden']:.2e}",
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
    loss, metrics = MSE(), [BinaryAccuracy(0.5)]

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
