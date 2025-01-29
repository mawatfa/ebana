from .equiprop_dc import EqPropDC


class EqProp:
    def __init__(self, model):
        self.model = model

        if self.model.simulation_kind == "op":
            self.alg = EqPropDC(self.model)

        elif self.model.simulation_kind == "tran":
            raise NotImplementedError("Not Implemented Yet !")

    def train(self, X, Y, batch_size, process_num, N_PROCESSES):
        return self.alg.train(X, Y, batch_size, process_num, N_PROCESSES)

    def evaluate(self, X, Y, batch_size, process_num, N_PROCESSES):
        return self.alg.evaluate(X, Y, batch_size, process_num, N_PROCESSES)

    def predict(self, X, Y, batch_size, process_num, N_PROCESSES):
        return self.alg.predict(X, Y, batch_size, process_num, N_PROCESSES)
