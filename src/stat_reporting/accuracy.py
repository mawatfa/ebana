import numpy as np

class AccuracyReporter:
    def __init__(self, model):
        self.model = model

    def measure_accuracy(self, dataset, prediction):
        target = dataset.outputs
        accuracy = {}
        for k in self.model.output_nodes.keys():
            c = self.model.loss_fn.verify_result(target[k], prediction[k])
            accuracy[k] = np.round(np.mean(c), 2)
        return accuracy

    def print_accuracy(self, accuracy, verbose=False):
        if verbose:
            print("\n-->Accuracy<--")
            for k in self.model.output_nodes.keys():
                print(f"    {k}: {accuracy[k]*100:.2f}%")
            print("")
