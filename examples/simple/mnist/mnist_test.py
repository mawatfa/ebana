import sys
from mnist import train

# run training with log <<<
def run_training_with_logging(config, index):
    log_file = f"training_log_{index}.log"
    with open(log_file, "w") as log:
        try:
            # Create a dual stream for stdout and stderr
            class DualStream:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, message):
                    for stream in self.streams:
                        stream.write(message)
                        stream.flush()

                def flush(self):
                    for stream in self.streams:
                        stream.flush()

            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = DualStream(sys.stdout, log)
            sys.stderr = DualStream(sys.stderr, log)

            result = train(config)
            print(f"Completed setup_training with config index {index}: {config}\nResult: {result}")
        except Exception as e:
            print(f"Error with config index {index}: {config}\nError: {e}")
        finally:
            # Restore original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
# >>>

# Base configuration
base_config = {
    "evaluate_only": False,
    "write_netlist": False,
    "n_processes_train": 25,
    "n_processes_test": 25,
    "dataset_size": 100,
    "train_batch_size": 25,
    "test_batch_size": 100,
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
    "save_state": True,
    "load_state": False,
}

# Modified configuration
new_config = {
    **base_config,
}

run_training_with_logging(base_config, index=1)
