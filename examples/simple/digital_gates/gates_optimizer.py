"""
Tune gates.py with optuna: each trial trains with the suggested settings
(on top of ARGS in gates.py) and is scored by the mean validation loss of
its last 3 epochs.
"""

# imports <<<
import numpy as np
import optuna

from gates import train
# >>>

# search space <<<
def settings(trial):
    return {
        "beta": trial.suggest_float("beta", 1e-12, 1e-9, log=True),
        "lr_hidden": trial.suggest_float("lr_hidden", 1e-10, 1e-8, log=True),
        "lr_or": trial.suggest_float("lr_or", 1e-11, 1e-9, log=True),
        "lr_and": trial.suggest_float("lr_and", 1e-11, 1e-9, log=True),
        "lr_xor": trial.suggest_float("lr_xor", 1e-11, 1e-9, log=True),
    }
# >>>

# objective <<<
def objective(trial):
    history = train({**settings(trial), "epochs": 20, "save": False, "verbose": False})
    loss = np.nanmean(history.epoch_values("val_loss")[-3:])
    return loss if np.isfinite(loss) else float("inf")     # the run diverged
# >>>


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=40)
    print("best settings:", study.best_params)
    print("best loss:", study.best_value)
