import pickle

class HistoryLoader:
    def __init__(self, accumulator):
        self.accumulator = accumulator

    def save_history(self, filepath):
        history = {
            "training_loss": self.accumulator.total_loss,
            "test_accuracy": self.accumulator.test_accuracy,
            "injected_currents": self.accumulator.injected_currents,
            **self.accumulator.get_data(),
        }

        with open(filepath, "wb") as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_history(self, filepath):
        with open(filepath, "rb") as handle:
            history = pickle.load(handle)

        self.total_loss = history["training_loss"]
        self.test_accuracy = history["test_accuracy"]
