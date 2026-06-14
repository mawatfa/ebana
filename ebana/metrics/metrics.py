class Metrics:
    """
    This class provides the means of ...
    """

    def __init__(
        self,
        model,
        save_phase_data=False,
        save_batch_data=False,
        save_injected_currents=False,
        verbose=True,
        validate_every=None,
    ):
        self.model = model
        self.save_phase_data = save_phase_data
        self.save_batch_data = save_batch_data
        self.save_injected_currents = save_injected_currents
        self.verbose = verbose
        self.validate_every = validate_every
