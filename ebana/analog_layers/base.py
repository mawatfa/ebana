from __future__ import annotations
from typing import Callable, List, Optional, Union, NoReturn
from abc import ABC, abstractmethod

class BaseLayer(ABC):
    """
    Abstract base for all layers.
    Defines common attributes and methods for building and connecting layers in a computational graph.
    """

    def __init__(
        self,
        name: str,
        units: int = -1,
        trainable: bool = False,
        save_sim_data: bool = False,
        grad_func: Optional[Callable] = None,
        weight_update_func: Optional[Callable] = None,
    ) -> None:
        self.name = name.lower()
        self.units = units
        self.trainable = trainable
        self.save_sim_data = save_sim_data
        self.built = False

        # TODO: Might need to change this because I may need to save only
        # the data required for training, and not everything.
        if trainable:
            self.save_sim_data = True

        # Hooks for layer-specific gradient and weight update behaviors.
        self.grad_func = grad_func if grad_func else self.default_grad_func
        self.weight_update_func = weight_update_func if weight_update_func else self.default_weight_update_func

        # Graph connections
        self.parent: Union[BaseLayer, List[BaseLayer], None] = None
        self.children: List[BaseLayer] = []

        # Shapes
        self.input_shape = None
        self.output_shape = None
        self.shape = None

        # Net names for internal and external connections
        ## External connections
        self.in_net_names: List[str] = []
        self.out_net_names: List[str] = []
        ## Internal connections
        self.in_subcircuit_net_names: List[str] = []
        self.out_subcircuit_net_names: List[str] = []

        # Branch names for measuring currents
        self.branch_names: List[str] = []

        # Storage for simulated data across phases
        self.voltages = {}
        self.currents = {}

    def __call__(self, inputs: BaseLayer | List[BaseLayer]) -> BaseLayer:
        """
        Makes the layer callable to enable chaining in a functional style.
        Ensures build is invoked once.
        """
        if not self.built:
            self.build(inputs)
            self.built = True
        return self

    def build(self, inputs: Optional[BaseLayer | List[BaseLayer]] = None) -> None:
        """
        Connects parent(s) and defines internal shapes, nets, branches, and other variables.
        """
        self._connect_parents(inputs)
        self._define_shapes()
        self._define_internal_nets()
        self._define_external_nets()
        self._define_internal_branches()
        self._build()

    def _connect_parents(self, inputs: Optional[BaseLayer | List[BaseLayer]]) -> None:
        """
        Registers the parent(s) of the current layer and notifies the parents of this layer as a child.
        """
        if inputs is None:
            return

        self.parent = inputs

        if isinstance(inputs, list):
            for parent_layer in inputs:
                parent_layer.children.append(self)
        else:
            inputs.children.append(self)

    @abstractmethod
    def _define_shapes(self) -> None:
        """
        Defines input_shape, output_shape, and shape for the layer.
        Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def _define_internal_nets(self) -> None:
        """
        Generates net names for internal subcircuits.
        Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def _define_external_nets(self) -> None:
        """
        Generates net names for external connections.
        Must be implemented by child classes.
        """
        pass

    def _define_internal_branches(self) -> None:
        """
        Generates internal branch names for measuring currents.
        Override in child classes if needed.
        """
        pass

    def _build(self) -> None:
        """
        Additional initialization can be performed here.
        Override in child classes if needed.
        """
        pass

    @abstractmethod
    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index: int) -> None:
        """
        Defines how the subcircuit of the layer is built in a SPICE-like environment.
        Must be implemented by child classes.
        """
        pass

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        """
        Returns the dimensions of the data to be saved at the end of a phase.
        Override as needed in child classes.
        """
        if not self.save_sim_data:
            return {}
        raise NotImplementedError("Override get_phase_data_spec if save_sim_data=True.")

    def get_phase_data(self) -> dict:
        """
        Returns the phase variable where the phase data is stored.
        """
        raise NotImplementedError("Override get_phase_data if save_sim_data=True.")

    def store_phase_data(self, sim_obj, phase: str) -> None:
        """
        Stores the simulation result for a particular phase.
        Override as needed in child classes.
        """
        if not self.save_sim_data:
            return
        raise NotImplementedError("Override store_phase_data if save_sim_data=True.")

    def get_batch_data_spec(self) -> dict:
        """
        Returns the dimensions of the data to be saved at the end of a batch.
        Override as needed in child classes.
        """
        if not self.save_sim_data:
            return {}
        raise NotImplementedError("Override get_batch_data_spec if save_sim_data=True.")

    def get_batch_data(self) -> dict:
        """
        Returns the variables where batch data is stored.
        """
        raise NotImplementedError("Override get_batch_data if save_sim_data=True.")

    def get_variables(self) -> dict:
        """
        Returns the state dictionary to be saved in the optimizer.
        Override as needed in child classes.
        """
        return {}

    def set_variables(self, optimizer_state) -> None:
        """
        Restore state of internal variables from the saved state.
        Override as needed in child classes.
        """
        pass

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase) -> NoReturn:
        """
        Specifies the data to be returned for use in the calculation of the gradient.
        Override as needed in child classes.
        """
        raise NotImplementedError("get_training_data must be implemented in the child class.")

    def default_grad_func(self, free, nudge) -> NoReturn:
        """
        Calculates and returns the gradient using the phase data.
        Override as needed in child classes.
        """
        raise NotImplementedError("default_grad_func must be implemented in the child class.")

    def default_weight_update_func(self, gradient, epoch_num, batch_num, num_batches) -> None:
        """
        Specifies how the internal weight variable is modified using the gradient.
        Override as needed in child classes.
        """
        raise NotImplementedError("default_weight_update_func must be implemented in the child class.")
