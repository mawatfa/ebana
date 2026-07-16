"""
Train mnist.py with several configurations in a row. Each run logs to
runs/<its name>.log; a run that fails is logged and the next one starts.
"""

# imports <<<
import logging

from mnist import ARGS, run_file, train
# >>>

# configurations <<<
# each one updates ARGS in mnist.py
CONFIGS = [
    {},                                                  # the defaults
    {"loss": "mse"},
    {"algorithm": "symmetric_ep"},
    {"hidden_units": 100, "lr_schedule": "cosine", "record_every": 25},
]
# >>>

# running <<<
def run(config):
    args = {**ARGS, **config}
    handler = logging.FileHandler(run_file(args, ".log"), mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        logging.info(f"config: {config}")
        history = train(config)
        logging.info("final: " + "  ".join(f"{k} {v:.4g}" for k, v in history.epochs[-1].items()))
    except Exception:
        logging.exception(f"run failed: {config}")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
# >>>


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for config in CONFIGS:
        run(config)
