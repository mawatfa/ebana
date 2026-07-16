"""
Train gates.py with several configurations in a row. Each run logs to
runs/<its name>.log; a run that fails is logged and the next one starts.
"""

# imports <<<
import logging

from gates import ARGS, run_file, train
# >>>

# configurations <<<
# each one updates ARGS in gates.py
CONFIGS = [
    {},                                                  # the defaults
    {"algorithm": "symmetric_ep"},
    {"optimizer": "momentum", "lr_schedule": "cosine"},
    {"update_rule": "power", "record_every": 5},
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
