import math
import numpy as np
from .locallyconnected2d import LocallyConnected2DLayer


class AveragePooling2DLayer(LocallyConnected2DLayer):
    """
    An AveragePooling2D layer implemented as a special case of LocallyConnected2D.
    In analog form, we set each patch's DenseLayer weights to the same value.
    """

    def __init__(
        self,
        name: str,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        initializer=None,
        lr: float = 0.0,
        trainable: bool = True,
        save_sim_data: bool = True,
        grad_func=None,
        weight_update_func=None,
    ):
        """
        Args:
            name: Layer name.
            pool_size: (pool_height, pool_width).
            strides: (stride_height, stride_width). Defaults to pool_size if not provided.
            padding: "valid" or "same".
            initializer: An object with init_high and init_low, used to pick the average weight.
            lr: Learning rate.
            trainable: Whether this layer is trainable.
            save_sim_data: Whether to save simulation data.
            grad_func: Custom gradient function.
            weight_update_func: Custom weight update function.
        """
        if strides is None:
            strides = pool_size  # match Keras behavior

        # For average pooling, number of filters usually matches input channels.
        # We'll let it match input channels later in _define_shapes.
        # For now, we set filters to 1, but we can update once we know input_shape.
        filters = 1

        # use LocallyConnected2D with kernel_size=pool_size, etc.
        super().__init__(
            name=name,
            kernel_size=pool_size,
            stride=strides,
            padding=padding,
            filters=filters,
            initializer=initializer,
            lr=lr,
            trainable=trainable,
            save_sim_data=save_sim_data,
            grad_func=grad_func,
            weight_update_func=weight_update_func,
        )

    def _define_shapes(self) -> None:
        """
        For average pooling, output channels match the input channels.
        We override this after LocallyConnected2D does its shape logic.
        """
        super()._define_shapes()

        # Match output channels to input channels
        in_height, in_width, in_channels = self.input_shape
        oh, ow, _ = self.output_shape
        self.output_shape = (oh, ow, in_channels)
        # Each patch's shape => (kh*kw*in_channels, in_channels)
        kh, kw = self.kernel_size
        self.shape = (oh, ow, kh * kw * in_channels, in_channels)
        self.num_patches = oh * ow
        self.filters = in_channels

    def _define_internal_nets(self) -> None:
        """
        Same as parent, but we rely on the updated output_shape and filter count.
        """
        super()._define_internal_nets()

    def _build(self):
        """
        Build each patch's DenseLayer, then force all weights to the same value
        based on the average of init_high and init_low.
        """
        # Let the normal build happen first
        for patch_layer in self.local_dense_layers:
            patch_layer._build()

        # Force each weight to the same value
        # (could be 1/(pool_size) if you prefer strict average, or simply an average of init_high/low)
        if self.initializer is not None:
            w = (self.initializer.init_high + self.initializer.init_low) / 2
        else:
            w = 1e-20  # fallback if no initializer is given

        for patch_layer in self.local_dense_layers:
            patch_layer.w = patch_layer.w * 0 + w
