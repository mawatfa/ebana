import sys
from gates import train

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
    "n_processes_train": 4,
    "n_processes_test": 4,
    "lr1": 5.18e-10,
    "lr2or": 4.54e-11,
    "lr2and": 2.32e-11,
    "lr2xor": 2.38e-11,
    "layer_1_trainable": True,
    "layer_2_trainable": True,
    "optimizer": "sgd",
    "loss": "mse",
    "num_epoch": 10,
    "validate_every_epoch": 1,
    "save_state": False,
    "load_state": False,
}

# Modified configuration
new_config = {
    **base_config,
}

run_training_with_logging(new_config, index=1)
