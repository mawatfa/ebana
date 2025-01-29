import optuna
from iris import train

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    config = {
        "beta": trial.suggest_float("beta", 1e-13, 1e-8, log=True),
        "lr1": trial.suggest_float("lr1", 1e-13, 1e-8, log=True),
        "lr2": trial.suggest_float("lr2", 1e-13, 1e-8, log=True),
    }

    # Base configuration
    base_config = {
        "evaluate_only": False,
        "write_netlist": False,
        "n_processes_train": 8,
        "n_processes_test": 8,
        "dataset_size": 105,
        "train_batch_size": 35,
        "test_batch_size": 45,
        "test_all": True,
        "beta": 1e-9,
        "lr1": 1e-10,
        "lr2": 1e-11,
        "layer_1_trainable": True,
        "layer_2_trainable": True,
        "optimizer": "sgd",
        "loss": "mse",
        "update_rule": "ep_sq",
        "quantization_bits": 2,
        "quantization_scale": 0.6,
        "num_epoch": 5,
        "validate_every_epoch": 1,
        "save_state": False,
        "load_state": False,
        "verbose": False,
        }

    # Merge base_config with trial configuration
    base_config.update(config)

    accuracy, loss = train(base_config)

    loss = loss["c1"][-3:-1]
    accuracy = accuracy["c1"][-3:-1]

    return loss.sum(), accuracy.sum()

# Create and optimize the study
study = optuna.create_study(directions=["minimize", "maximize"])

study.optimize(objective, n_trials=50, n_jobs=5)

# Output the Pareto front
print("")
print("Number of Pareto optimal solutions:", len(study.best_trials))
for trial in study.best_trials:
    print("Trial values:", trial.values)
    print("Params:", trial.params)
