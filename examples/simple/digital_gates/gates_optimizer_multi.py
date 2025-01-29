import optuna
from gates import train

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    config = {
        "beta": trial.suggest_float("beta", 1e-11, 1e-7, log=True),
        "lr1": trial.suggest_float("lr1", 1e-11, 1e-7, log=True),
        "lr2or": trial.suggest_float("lr2or", 1e-11, 1e-7, log=True),
        "lr2and": trial.suggest_float("lr2and", 1e-11, 1e-7, log=True),
        "lr2xor": trial.suggest_float("lr2xor", 1e-11, 1e-7, log=True),
    }

    # Base configuration
    base_config = {
        "evaluate_only": False,
        "write_netlist": False,
        "n_processes_train": 4,
        "n_processes_test": 4,
        "beta": 1e-7,
        "lr1": 4e-8,
        "lr2or": 4e-9,
        "lr2and": 4e-9,
        "lr2xor": 4e-9,
        "layer_1_trainable": True,
        "layer_2_trainable": True,
        "optimizer": "sgd",
        "loss": "mse",
        "num_epoch": 20,
        "validate_every_epoch": 1,
        "save_state": False,
        "load_state": False,
        "verbose": False,
    }

    # Merge base_config with trial configuration
    base_config.update(config)

    accuracy, loss = train(base_config)

    loss_or = loss["or"][-3:-1]
    loss_and = loss["and"][-3:-1]
    loss_xor = loss["xor"][-3:-1]

    return loss_or.sum(), loss_and.sum(), loss_xor.sum()

# Create and optimize the study
study = optuna.create_study(directions=["minimize", "minimize", "minimize"])

study.optimize(objective, n_trials=40, n_jobs=10)

# Output the Pareto front
print("")
print("Number of Pareto optimal solutions:", len(study.best_trials))
for trial in study.best_trials:
    print("Trial values:", trial.values)
    print("Params:", trial.params)
