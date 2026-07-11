"""What a Recorder measures. Any picklable probe(model, sim) returning an array works."""


class Voltages:
    """The voltages of a layer's output nets."""

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, model, sim):
        return sim.voltages(model.layer(self.layer).output)


class Measurement:
    """What the device of a parameter measures, e.g. Measurement("d1/weight")."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __call__(self, model, sim):
        return model.parameter(self.parameter).measure(sim)


class Call:
    """A layer method called with the simulator, e.g. Call("meter", "currents")."""

    def __init__(self, layer, method):
        self.layer, self.method = layer, method

    def __call__(self, model, sim):
        return getattr(model.layer(self.layer), self.method)(sim)
