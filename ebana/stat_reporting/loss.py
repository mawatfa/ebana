class LossReporter:
    def __init__(self, model, accumulator):
        self.model = model
        self.accumulator = accumulator

    def print_iteration_loss(self, epoch_num, batch_num):
        losses = {k: v[epoch_num][batch_num] for k, v in self.accumulator.total_loss.items()}
        print(f"epoch {epoch_num + 1} batch {batch_num + 1}: " + ", ".join(f"{k} {v:.6g}" for k, v in losses.items()))

    def print_epoch_loss(self, epoch_num):
        losses = {k: v[epoch_num].mean() for k, v in self.accumulator.total_loss.items()}
        print(f"epoch {epoch_num + 1} loss: " + ", ".join(f"{k} {v:.6g}" for k, v in losses.items()))
