import numpy as np
import math
from .dense_layers import DenseLayer
from ..two_terminal_devices import TwoTerminals
from ...utils.generating_functions import generate_names_from_shape
from ...schedules import LearningRateSchedule


class LocallyConnected2DLayer(TwoTerminals):
    """
    A 2D locally-connected layer to mimic the convolution operation.
    """
    def __init__(
        self,
        name:str,
        kernel_size:tuple,
        stride:tuple,
        padding:str,
        filters:int,
        initializer=None,
        lr:float|LearningRateSchedule=0.0,
        trainable:bool=True,
        save_sim_data:bool=True,
        grad_func=None,
        weight_update_func=None,
    ):
        """
        Args:
            name: Name of the layer.
            kernel_size: (kernel_height, kernel_width).
            stride: (stride_height, stride_width).
            padding: "valid" or "same"
            filters: Number of output channels per local patch.
            initializer: Object with a method `initialize_weights(shape)`.
            lr: Learning rate.
            trainable: Whether this layer is trainable.
            save_sim_data: Whether to save simulation data.
            grad_func: Custom gradient function if desired.
            weight_update_func: Custom weight update function if desired.
        """
        super().__init__(
            name=name,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_update_func,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filters = filters
        self.initializer = initializer
        self.lr = lr
        self.paddded_net_name = f"{self.name}_pad"

        # we'll hold one DenseLayer per local patch
        self.local_dense_layers = []
        # also store per-patch info
        self.patch_info = {}

    def get_padding_dimensions(self) -> tuple[int, int]:
        if self.padding == "same":
            in_height, in_width, in_channels = self.input_shape # type: ignore
            sh, sw = self.stride
            kh, kw = self.kernel_size
            # for each dimension, pad so output_height = ceil(in_height/stride).
            out_height = math.ceil(in_height / sh)
            out_width = math.ceil(in_width / sw)
            # total padding needed
            total_pad_h = max((out_height - 1) * sh + kh - in_height, 0)
            total_pad_w = max((out_width - 1) * sw + kw - in_width, 0)
            return (total_pad_h // 2, total_pad_w // 2)
        else:
            return (0, 0)

    def _define_shapes(self) -> None:
        """
        The parent's output_shape should already be (height, width, channels).
        We compute the LocallyConnected2D's output shape (out_height, out_width, filters).
        """
        if isinstance(self.parent, list):
            raise ValueError("LocallyConnected2D layer supports only one parent.")

        self.input_shape = self.parent.output_shape         # type: ignore
        in_height, in_width, in_channels = self.input_shape # type: ignore

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.get_padding_dimensions()

        # compute output height/width
        out_height = ((in_height + 2 * ph - kh) // sh) + 1
        out_width  = ((in_width  + 2 * pw - kw) // sw) + 1

        # number of patches
        self.num_patches = out_height * out_width

        # final shape: (out_height, out_width, filters)
        self.output_shape = (out_height, out_width, self.filters)

        # each patch is a dense transform with shape = (kh * kw * in_channels, filters).
        self.shape = (out_height, out_width, kh * kw * in_channels, self.filters)

    def _define_internal_nets(self) -> None:
        """
        1) We expect parent.out_net_names to be a 3D array of shape (in_height, in_width, in_channels).
        2) Pad the input array
        2) Extract local_in_nets for each patch.
        3) Assign out_net_names for the entire LocallyConnected2D layer.
        4) Instantiate a DenseLayer for each patch.
        """
        # original input shape
        in_height, in_width, in_channels = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.get_padding_dimensions()
        out_height, out_width, _ = self.output_shape

        # TODO: I may need to make a deep copy of this first
        in_net_names = self.parent.out_net_names

        # pad the array with padding name
        padded_in_nets = np.pad(
            in_net_names,
            pad_width=((ph, ph), (pw, pw), (0, 0)),
            mode="constant",
            constant_values=self.paddded_net_name
        )

        # generate out_net_names
        self.out_net_names = generate_names_from_shape(self.output_shape, f"n_{self.name}")

        # build each local patch
        for oh in range(out_height):
            for ow in range(out_width):
                row_start = oh * sh
                col_start = ow * sw

                # Slice and flatten patch
                patch_in_nets = (
                    padded_in_nets[
                        row_start : row_start + kh,
                        col_start : col_start + kw,
                        :
                    ]
                    .ravel()
                    .tolist()
                )

                patch_out_nets = self.out_net_names[oh][ow]

                # Create DenseLayer
                patch_layer = DenseLayer(
                    name=f"{self.name}_patch_{oh}_{ow}",
                    units=self.filters,
                    initializer=self.initializer,
                    lr=self.lr,
                    trainable=self.trainable,
                    save_sim_data=self.save_sim_data,
                    grad_func=self.grad_func,
                    weight_upata_func=self.weight_update_func,
                )
                patch_layer.input_shape = (len(patch_in_nets),)
                patch_layer.output_shape = (self.filters,)
                patch_layer.shape = (len(patch_in_nets), self.filters)

                # internal and external nets
                patch_layer.in_subcircuit_net_names = generate_names_from_shape(patch_layer.input_shape, "IN")
                patch_layer.out_subcircuit_net_names = generate_names_from_shape(patch_layer.output_shape, "OUT")
                patch_layer.in_net_names = patch_in_nets
                patch_layer.out_net_names = patch_out_nets

                # track patch info
                self.local_dense_layers.append(patch_layer)
                self.patch_info[(oh, ow)] = {
                    "dense_layer": patch_layer,
                    "in_names": patch_in_nets,
                    "out_names": patch_out_nets,
                }

    def _define_external_nets(self) -> None:
        # We've already defined self.out_net_names in _define_internal_nets().
        pass

    def _build(self):
        """
        Build each patch's DenseLayer (initialize weights, etc.).
        """
        for patch_layer in self.local_dense_layers:
            patch_layer._build()

    def build_spice_subcircuit(self, circuit, spice_input_dict, sample_index: int) -> None:
        """
        Build each patch's subcircuit in the overall circuit.
        """
        for patch_layer in self.local_dense_layers:
            patch_layer.build_spice_subcircuit(circuit, spice_input_dict, sample_index)

    # -------------------------------------------------
    # Saving methods (only matter if save_sim_data=True)
    # -------------------------------------------------
    def get_phase_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        specs = {}
        for (oh, ow), info in self.patch_info.items():
            specs[f"patch_{oh}_{ow}"] = info["dense_layer"].get_phase_data_spec()
        return {}

    def get_batch_data_spec(self) -> dict:
        if not self.save_sim_data:
            return {}
        specs = {}
        for (oh, ow), info in self.patch_info.items():
            specs[f"patch_{oh}_{ow}"] = info["dense_layer"].get_batch_data_spec()
        return {}

    def store_phase_data(self, sim_obj, phase):
        if self.save_sim_data:
            for patch_layer in self.local_dense_layers:
                patch_layer.store_phase_data(sim_obj, phase)

    def get_phase_data(self):
        if not self.save_sim_data:
            return {}
        data = {}
        for patch_layer in self.local_dense_layers:
            data[patch_layer.name] = patch_layer.get_phase_data()["voltages"]
        return {}

    def get_batch_data(self):
        if not self.save_sim_data:
            return {}
        data = {}
        for patch_layer in self.local_dense_layers:
            data[patch_layer.name] = patch_layer.get_batch_data()
        return {}

    def get_variables(self) -> dict:
        """
        Gather the variables (e.g., weights) from each patch's DenseLayer.
        """
        vars_ = {}
        for patch_layer in self.local_dense_layers:
            vars_[patch_layer.name] = patch_layer.get_variables()
        return vars_

    def set_variables(self, optimizer_state):
        """
        Gather the variables (e.g., weights) from each patch's DenseLayer.
        """
        for patch_layer in self.local_dense_layers:
            patch_layer.set_variables(optimizer_state[patch_layer.name])

    # -------------------------------------------------
    # Training methods (only matter if trainable=True)
    # -------------------------------------------------
    def get_training_data(self, phase):
        """
        Collect training data from each patch (e.g. voltage 'drop').
        """
        data = np.zeros(self.shape)
        for (oh, ow), info in self.patch_info.items():
            data[oh, ow] += info["dense_layer"].get_training_data(phase)
        return data

    def default_weight_update_func(self, gradient, epoch_num, batch_num, num_batches):
        for (oh, ow), info in self.patch_info.items():
            info["dense_layer"].default_weight_update_func(gradient[oh, ow], epoch_num, batch_num, num_batches)


