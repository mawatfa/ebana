# imports <<<
import numpy as np
from ..simulation import Simulator
from ..data_collection.process_data_accumulator import ProcessDataAccumulator
# >>>

class EqPropDC:

    # constructor <<<
    def __init__(self, model):
        self.model = model
        self.sim = Simulator()
        self.default_free_phase_name = "free"
        self.default_nudge_phase_name = "nudge"
    # >>>

    # default methods <<<
    def default_ep_nudging_phases(self, X, Y, idx, injected_current):
        self.nudging_phase(X, Y, idx, injected_current, self.default_nudge_phase_name)

    def default_ep_gradient_calculation(self):
        free = self.measurements[self.default_free_phase_name]
        nudge = self.measurements[self.default_nudge_phase_name]
        for p in self.model.trainable_parameters():
            p.grad += p.gradient(free[p.full_name], nudge[p.full_name])
    # >>>

    # free phase <<<
    def free_phase(self, X, Y, idx, training=True):
        """
        This implements the free phase of the EqProp algorithm,
        where no current is injected into the output nodes.
        """
        # Only one sample is simulated at a time
        self.sim.load(self.model.build_netlist(X, idx))
        self.sim.solve()

        # Get simulation data
        if training:
            self.store_phase_data(self.default_free_phase_name)

        # Get prediction of the model
        self.network_prediction = {}
        for layer in self.model.outputs:
            self.network_prediction[layer.name] = layer.get_layer_spice_output(self.sim)
    # >>>

    # nudging phase <<<
    def nudging_phase(self, X, Y, idx, injected_current, phase):
        """
        This implements the nudging phase of the EqProp algorithm,
        where loss gradient current is injected into the output nodes.
        """
        # In the nudging phase, we inject a current into each output node
        # of the already-loaded circuit, and simulate it again.
        for layer in self.model.outputs:
            self.sim.alter(*layer.nudge(injected_current[layer.name]))
        self.sim.solve()

        # Get the simulation data again
        self.store_phase_data(phase)

    def store_phase_data(self, phase):
        """Measure every trainable parameter after this phase's solve."""
        self.measurements[phase] = {p.full_name: p.measure(self.sim) for p in self.model.trainable_parameters()}
    # >>>

    # train <<<
    def train(self, X, Y, batch_size, process_num, N_PROCESSES):
        """
        Training loop according to the equilibirum propagation algorithm.
        """
        # Gradients accumulate over this process's samples.
        for p in self.model.trainable_parameters():
            p.grad = np.zeros(p.shape)
        self.measurements = {}

        # Initialize metrics variables in case they should be saved
        self.data_accumulator = ProcessDataAccumulator(self.model, batch_size)

        # Maintain a running sum of the loss of each sample in the dataset
        batch_loss = {}
        for k, v in self.model.output_nodes.items():
            batch_loss[k] = np.zeros(shape=(v,))

        # Initalize currents to be injected to zero
        injected_currents = {}
        if self.model.metrics.save_injected_currents:
            for k, v in self.model.output_nodes.items():
                injected_currents[k] = np.zeros(shape=(batch_size, v))

        #######################################################################
        #                           Algorithm Begin                           #
        #######################################################################
        # each process handles ~ (batch_size / N_PROCESSES) samples
        for idx in range(process_num, batch_size, N_PROCESSES):
            # Run free phase for the current sample.
            self.free_phase(X, Y, idx)

            # Save the metrics after the free phase.
            self.data_accumulator.save_metrics_during_training(idx, self.default_free_phase_name, self.sim)

            # Calculate the loss during the free phase and the current that must be injected during the nudging phase.
            injected_current, losses = {}, {}
            for layer in self.model.outputs:
                layer_output = layer.calculate_layer_output(self.network_prediction[layer.name])
                layer_prediction = layer.calculate_layer_prediction(layer_output)
                losses[layer.name], injected_current[layer.name] = self.model.loss_fn(layer_prediction, Y[layer.name][idx])

            # Add loss of the sample to batch loss.
            for k, v in self.model.output_nodes.items():
                batch_loss[k] += losses[k]

            if self.model.metrics.save_injected_currents:
                for k, v in self.model.output_nodes.items():
                    injected_currents[k][idx] = injected_current[k]

            # Run nudging phase for the current sample.
            self.default_ep_nudging_phases(X, Y, idx, injected_current)

            # Save the metrics after the free phase.
            self.data_accumulator.save_metrics_during_training(idx, self.default_nudge_phase_name, self.sim)

            # Calculate the update according to the EP algorithm
            self.default_ep_gradient_calculation()

        #######################################################################
        #                            Algorithm End                            #
        #######################################################################
        # At the end of the batch, retrieve all the necessary variables.
        return {
            "process_layer_collected_data": self.data_accumulator.get_data(),
            "process_parameter_gradients": {p.full_name: p.grad for p in self.model.trainable_parameters()},
            "process_injected_currents": injected_currents,
            "process_losses": batch_loss,
        }

    # >>>

    # evaluate <<<
    def evaluate(self, X, Y, batch_size, process_num, N_PROCESSES):
        """
        Training loop according to the equilibirum propagation algorithm.
        """

        process_predictions = {}
        batch_loss = {}
        for k, v in self.model.output_nodes.items():
            process_predictions[k] = np.zeros(shape=(batch_size, v))
            batch_loss[k] = np.zeros(shape=(v,))

        # each process handles ~ (batch_size / N_PROCESSES) samples
        for idx in range(process_num, batch_size, N_PROCESSES):
            # Run free phase for the current sample.
            self.free_phase(X, Y, idx, training=False)

            # Get network prediction
            losses = {}
            for layer in self.model.outputs:
                layer_output = layer.calculate_layer_output(self.network_prediction[layer.name])
                layer_prediction = layer.calculate_layer_prediction(layer_output)
                process_predictions[layer.name][idx] = layer_prediction
                losses[layer.name], _ = self.model.loss_fn(layer_prediction, Y[layer.name][idx])

            # Add loss of the sample to batch loss.
            for k, v in self.model.output_nodes.items():
                batch_loss[k] += losses[k]

        return {
            "process_predictions": process_predictions,
            "process_losses": batch_loss,
        }

    # >>>

    # predict <<<
    def predict(self, X, Y, batch_size, process_num, N_PROCESSES):
        """
        Training loop according to the equilibirum propagation algorithm.
        """

        process_output = {}
        process_prediction = {}
        for k, v in self.model.output_nodes.items():
            process_output[k] = np.zeros(shape=(batch_size, v))
            process_prediction[k] = np.zeros(shape=(batch_size, v))

        # each process handles ~ (batch_size / N_PROCESSES) samples
        for idx in range(process_num, batch_size, N_PROCESSES):

            # Run free phase for the current sample.
            self.free_phase(X, Y, idx, training=False)

            for layer in self.model.outputs:
                layer_output = layer.calculate_layer_output(self.network_prediction[layer.name])
                layer_prediction = layer.calculate_layer_prediction(layer_output)
                process_output[layer.name][idx] = layer_output
                process_prediction[layer.name][idx] = layer_prediction

        return {
            "process_output": process_output,
            "process_prediction": process_prediction,
        }

    # >>>
