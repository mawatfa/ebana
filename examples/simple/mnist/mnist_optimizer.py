import optuna
from mnist import train

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    config = {
        "lr1": trial.suggest_float("lr1", 1e-8, 1e-5, log=True),
        "lr2": trial.suggest_float("lr2", 1e-8, 1e-5, log=True),
        "beta": trial.suggest_float("beta", 1e-8, 1e-5, log=True),
    }

    # Base configuration
    base_config = {
        "evaluate_only": False,
        "write_netlist": False,
        "n_processes_train": 20,
        "n_processes_test": 20,
        "dataset_size": 120,
        "train_batch_size": 40,
        "test_batch_size": 40,
        "h1_units": 50,
        "beta": 1e-6,
        "lr1": 1e-7,
        "lr2": 1e-7,
        "layer_1_trainable": True,
        "layer_2_trainable": True,
        "optimizer": "adam",
        "loss": "crossentropy",
        "update_rule": "ep_sq",
        "num_epoch": 5,
        "save_state": False,
        "load_state": False,
    }

    # Merge base_config with trial configuration
    base_config.update(config)

    accuracy, loss = train(base_config)

    loss = loss["c1"][-3:-1]
    accuracy = accuracy["c1"][-3:-1]

    return loss.sum()
    # return -accuracy.sum()

# Create and optimize the study
study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=10, n_jobs=2)

# Output the best parameters
print("Best hyperparameters:", study.best_params)

