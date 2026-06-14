
class TrainingValidator:
    def __init__(self, model, test_accuracy, metrics):
        self.model = model
        self.test_accuracy = test_accuracy
        self.metrics = metrics

    def evaluate_validation(self, dataloader, epoch_num, batch_num):
        if "epoch_num" in self.metrics.validate_every.keys():
            if ((epoch_num % self.metrics.validate_every["epoch_num"] == 0) and (batch_num == self.model.num_batches - 1)):
                dataset = next(dataloader)
                accuracy, _ = self.model.evaluate(dataset, verbose=self.metrics.verbose)
                for k in self.model.output_nodes.keys():
                    self.test_accuracy[k][epoch_num] = accuracy[k]

        elif "batch_num" in self.metrics.validate_every.keys():
            if ((batch_num + 1) % self.metrics.validate_every["batch_num"] == 0) or ((epoch_num == self.model.epochs - 1) and (batch_num == self.model.num_batches)):
                dataset = next(dataloader)
                accuracy, _ = self.model.evaluate(dataset, verbose=self.metrics.verbose)
                # TODO
                # for k in self.model.output_nodes.keys():
                #     self.test_accuracy[k][epoch_num][batch_num] = accuracy[k]
