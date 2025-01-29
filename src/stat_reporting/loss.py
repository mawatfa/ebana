import numpy as np

class LossReporter:
    def __init__(self, model, total_loss):
        self.model = model
        self.total_loss = total_loss

    def print_iteration_loss(self, epoch_num, batch_num):
        loss, loss_mean = {}, {}

        if self.model.dataset_size != self.model.batch_size:
            print(f"Iteration {epoch_num+1}/{batch_num+1}:")
            for k in self.model.output_nodes.keys():
                loss[k] = self.total_loss[k][epoch_num][batch_num] / self.model.batch_size
                loss_mean[k] = np.sum(loss[k])
        else:
            print(f"Iteration {epoch_num+1}:")
            for k in self.model.output_nodes.keys():
                loss[k] = self.total_loss[k][epoch_num] / self.model.dataset_size
                loss_mean[k] = np.sum(loss[k])

        for k in self.model.output_nodes.keys():
            print(f"    {k}: {loss_mean[k]:.7f} -> {loss[k]}")

    def print_total_loss(self):
        loss = 0
        for k in self.model.output_nodes.keys():
            loss += self.total_loss[k][self.model.epoch_num].sum()
        print("Total Epoch Loss: ", loss / self.model.batch_size)
        print("")
