import sys
from iris import train

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
            print(f"Completed setup_training with config index {index}")
            print(config)
            print(f"Accuracy: {result[0]['c1']}")
            print(f"Loss: {result[1]['c1']}")
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
    "n_processes_train": 8,
    "n_processes_test": 8,
    "dataset_size": 105,
    "train_batch_size": 35,
    "test_batch_size": 45,
    "test_all": True,
    "beta": 8e-9,
    "lr1": 9.6e-10,
    "lr2": 5.9e-11,
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
}

# Modified configuration
new_config = {
    **base_config,
}

run_training_with_logging(new_config, index=1)
