"""
Tune mnist.py with optuna: each trial trains with the suggested settings
(on top of ARGS in mnist.py) and is scored by the mean validation loss of
its last 2 epochs.
"""

# imports <<<
import numpy as np
import optuna

from mnist import train
# >>>

# search space <<<
def settings(trial):
    return {
        "beta": trial.suggest_float("beta", 1e-8, 1e-5, log=True),
        "lr_hidden": trial.suggest_float("lr_hidden", 1e-8, 1e-6, log=True),
        "lr_out": trial.suggest_float("lr_out", 1e-8, 1e-6, log=True),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
        "stride": trial.suggest_categorical("stride", [2, 3]),
    }
# >>>

# objective <<<
def objective(trial):
    history = train({**settings(trial), "samples": 200, "epochs": 3, "save": False, "verbose": False})
    loss = np.nanmean(history.epoch_values("val_loss")[-2:])
    return loss if np.isfinite(loss) else float("inf")     # the run diverged
# >>>


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=20)
    print("best settings:", study.best_params)
    print("best loss:", study.best_value)
