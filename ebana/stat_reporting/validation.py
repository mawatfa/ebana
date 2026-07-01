class TrainingValidator:
    def __init__(self, model, accumulator, monitor):
        self.model = model
        self.accumulator = accumulator
        self.monitor = monitor

    def evaluate_validation(self, dataset, epoch_num, batch_num):
        every = self.monitor.validate_every
        last_batch = batch_num == self.model.num_batches - 1
        if "epoch_num" in every:
            due = last_batch and epoch_num % every["epoch_num"] == 0
        else:
            due = (batch_num + 1) % every["batch_num"] == 0 or (last_batch and epoch_num == self.model.epochs - 1)
        if due:
            results = self.model.evaluate(dataset, verbose=self.monitor.verbose)
            if last_batch:
                self.accumulator.store_validation(results, epoch_num)
