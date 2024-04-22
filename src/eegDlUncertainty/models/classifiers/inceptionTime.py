"""
Inception Time is implemented. Defaults are set as the original Keras implementation.

Paper: https://arxiv.org/pdf/1909.04939.pdf
Original implementation in keras at https://github.com/hfawaz/InceptionTime

This implementation was authored by Thomas Tveitstøl (Oslo University Hospital)
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eegDlUncertainty.models.base_classifier import BaseClassifier


# ---------------------------
# Sub-modules
# ---------------------------
class _InceptionModule(nn.Module):

    num_kernel_sizes = 3

    def __init__(self, in_channels: int, units: int, use_bottleneck: bool , max_kernel_size: int,
                 mc_dropout_enabled, mc_dropout_rate, activation: Optional[Callable] = None):
        """
        Initialise

        As opposed to the original keras implementation, strides is strictly set to 1 and cannot be specified to any
        other value. This is because setting padding='same' is not supported when strides are greater than 1
        Args:
            in_channels: Number of expected input channels
            units: Output (channel) dimension of the Conv layers. Equivalent to nb_filters in original keras
                implementation
            activation: Activation function. If None is passed, no activation function will be used
            use_bottleneck: To use the first input_conv layer or not
            max_kernel_size: Largest kernel size used. In the original keras implementation, the equivalent argument is
                stored as kernel_size - 1, the same is not done here
        """
        super().__init__()
        # Store selected activation function
        self._activation_function = _no_activation_function if activation is None else activation

        # -------------------------------
        # Define Conv layer maybe operating on
        # the input
        # -------------------------------
        if use_bottleneck:
            self._input_conv = nn.Conv1d(in_channels, out_channels=32, kernel_size=1, padding="same", bias=False)
            out_channels = 32
        else:
            self._input_conv = None
            out_channels = in_channels

        # -------------------------------
        # Define convolutional layers with different
        # kernel sizes (to be concatenated at the end)
        # -------------------------------
        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(_InceptionModule.num_kernel_sizes)]
        # kernel_sizes = [9, 19, 39]

        self._conv_list = nn.ModuleList([nn.Conv1d(in_channels=out_channels, out_channels=units,
                                                   kernel_size=kernel_size, stride=1, padding="same", bias=False)
                                         for kernel_size in kernel_sizes])

        # -------------------------------
        # Define Max pooling and conv layer to be
        # applied after max pooling
        # -------------------------------
        self._max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self._conv_after_max_pool = nn.Conv1d(in_channels=in_channels, out_channels=units, kernel_size=1,
                                              padding="same", bias=False)
        # self._dropout = nn.Dropout(p=mc_dropout_rate)
        self._mc_dropout_enabled = mc_dropout_enabled

        if self._mc_dropout_enabled:
            self._dropout_list = nn.ModuleList([nn.Dropout(p=mc_dropout_rate) for _ in range(len(kernel_sizes))])

        # Finally, define batch norm
        self._batch_norm = nn.BatchNorm1d(num_features=units * (len(self._conv_list) + 1))  # Must multiply due to
        # concatenation with all outputs from self._conv_list and self._con_after_max_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: A torch.Tensor with shape=(batch, channels, time steps)

        Returns:
            Output of inception module
        Examples:
            >>> my_inception_module = _InceptionModule(in_channels=53)
            >>> my_data = torch.rand(size=(10, 53, 600))
            >>> my_inception_module(torch.rand(size=(10, 53, 600))).size()
            torch.Size([10, 128, 600])
            >>> # Verify that the code runs with different arguments:
            >>> _InceptionModule(in_channels=53, units=11, activation=F.relu, use_bottleneck=False,
            ...                  max_kernel_size=8)(my_data).size()
            torch.Size([10, 44, 600])
        """
        # Maybe pass through input conv
        if self._input_conv is not None:
            inception_input = self._activation_function(self._input_conv(x))
        else:
            inception_input = torch.clone(x)

        # Pass through the conv layers with different kernel sizes
        outputs = []
        for idx, conv_layer in enumerate(self._conv_list):
            conv_output = self._activation_function(conv_layer(inception_input))

            if self._mc_dropout_enabled:
                # conv_output = self._dropout(conv_output)
                conv_output = self._dropout_list[idx](conv_output)

            outputs.append(conv_output)

        # Pass input tensor through max pooling, followed by a conv layer
        max_pool_output = self._max_pool(x)
        outputs.append(self._activation_function(self._conv_after_max_pool(max_pool_output)))

        # Concatenate, add batch norm, apply Relu activation function and return
        x = torch.cat(outputs, dim=1)  # concatenate in channel dimension
        x = F.relu(self._batch_norm(x))

        return x


class _ShortcutLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialise
        Args:
            in_channels: Expected number of input channels
            out_channels: Expected number of channel of the tensor we want to add short layer output to (see Examples in
                forward method)
        """
        super().__init__()
        # Define Conv layer and batch norm
        self._conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same")
        self._batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            input_tensor: a torch.Tensor with shape=(batch, in_channels, time_steps), where in_channels is equal to what
                was passed to __init__
            output_tensor: a torch.Tensor with shape=(batch, out_channelsl, time_steps), where out_channels is equal to
                what was passed to __init__

        Returns:
            Output of shortcut layer
        Examples:
            >>> my_model = _ShortcutLayer(in_channels=43, out_channels=76)
            >>> my_model(input_tensor=torch.rand(size=(10, 43, 500)),
            ...          output_tensor=torch.rand(size=(10, 76, 500))).size()  # The size is the same as output_tensor
            torch.Size([10, 76, 500])
            >>> # Raises a RuntimeError if the tensors do not have expected shapes
            >>> my_model(input_tensor=torch.rand(size=(10, 43, 500)),
            ...          output_tensor=torch.rand(size=(10, 75, 500))).size()
            Traceback (most recent call last):
            ...
            RuntimeError: The size of tensor a (76) must match the size of tensor b (75) at non-singleton dimension 1
        """
        # Pass through conv layer and batch norm
        x = self._conv(input_tensor)
        x = self._batch_norm(x)

        # Add to output tensor, apply Relu and return
        return F.relu(x + output_tensor)


# ---------------------------
# Main module
# ---------------------------
class InceptionNetwork(BaseClassifier):

    activation_function = "linear"

    def __init__(self, **kwargs):
        """
        Initialise
        Keyword Args:
            in_channels: Expected number of input channels
            num_classes: Output dimension of prediction. That is, the output of the forward method will have
                shape=(batch, num_classes)
            cnn_units: Number of output channels of the Inception modules
            depth: Number of Inception modules used
            use_bottleneck: Using bottleneck or not
            activation: Activation function to use in Inception modules
            max_kernel_size: Max kernel size of in Inception modules
            use_residual: To use Shortcut layers or not
        """
        # -----------------------------
        # Required kwargs
        # -----------------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------------
        # Optional kwargs
        # -----------------------------
        cnn_units: int = kwargs.get("cnn_units", 32)
        depth: int = kwargs.get("depth", 6)
        use_bottleneck: bool = kwargs.get("use_bottleneck", True)
        activation: Optional[Callable] = kwargs.get("activation")
        max_kernel_size: int = kwargs.get("max_kernel_size", 40)
        use_residual: bool = kwargs.get("use_residual", True)
        mc_dropout_enabled: bool = kwargs.get("mc_dropout_enabled")
        mc_dropout_rate: float = kwargs.get("mc_dropout_rate")

        # -----------------------------
        # Store hyperparameters
        # -----------------------------
        super().__init__(in_channels=in_channels, num_classes=num_classes, cnn_units=cnn_units, depth=depth,
                         use_bottleneck=use_bottleneck, activation=activation, max_kernel_size=max_kernel_size,
                         use_residual=use_residual, classifier_name=kwargs.get("classifier_name"),
                         mc_dropout_enabled=mc_dropout_enabled, mc_dropout_rate=mc_dropout_rate)

        # -----------------------------
        # Define Inception modules
        # -----------------------------
        output_channels = cnn_units * (_InceptionModule.num_kernel_sizes + 1)  # Output channel dim of inception modules
        self._inception_modules = nn.ModuleList(
            [_InceptionModule(in_channels=in_channel, units=cnn_units,
                              use_bottleneck=use_bottleneck, activation=activation,
                              max_kernel_size=max_kernel_size, mc_dropout_enabled=mc_dropout_enabled,
                              mc_dropout_rate=mc_dropout_rate)
             for i, in_channel in enumerate([in_channels] + [output_channels]*(depth - 1))]
        )

        # -----------------------------
        # Define Shortcut layers
        # -----------------------------
        if use_residual:
            # Calculate how many shortcut layers are needed
            num_shortcut_layers = len(self._inception_modules) // 3
            self._shortcut_layers = nn.ModuleList()
            for i in range(num_shortcut_layers):
                in_channels = in_channels if i == 0 else output_channels
                self._shortcut_layers.append(
                    _ShortcutLayer(in_channels=in_channels, out_channels=output_channels)
                )
        else:
            self._shortcut_layers = None

        # -----------------------------
        # Define FC layer for output (global
        # average pooling is implemented in
        # forward method)
        # -----------------------------
        self._fc_layer = nn.Linear(in_features=output_channels,
                                   out_features=num_classes)
        self._dropout = nn.Dropout(p=0.3)
        self._fc_layer_age = nn.Linear(in_features=output_channels + 1, out_features=num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward method of Inception
        Args:
            input_tensor: A torch.Tensor with shape=(batch, channels, time steps)

        Returns:
            Predictions, without activation function. No activation function is used, as it is often more numerically
            stable to use loss function with logits
        """
        if input_tensor.shape[1] == 20:
            eeg_data = input_tensor[:, :-1, :]
            # Only select one of the expanded ages in channel 2...
            age = input_tensor[:, -1, 0].unsqueeze(1)  #
        else:
            eeg_data = input_tensor
            age = None

        x = torch.clone(eeg_data)
        input_res = torch.clone(eeg_data)

        # Make shortcut layers iterable, if not None
        shortcut_layers = None if self._shortcut_layers is None else iter(self._shortcut_layers)

        for i, inception_module in enumerate(self._inception_modules):
            # Pass though Inception module
            x = inception_module(x)

            # # If shortcut layers are included, use them for every third inception module
            if shortcut_layers is not None and i % 3 == 2:
                shortcut_layer = next(shortcut_layers)
                x = shortcut_layer(input_tensor=input_res, output_tensor=x)
                input_res = torch.clone(x)

        # Global Average Pooling in time dimension. Note that this operation allows a varied numer of time steps to be
        # used
        x = torch.mean(x, dim=-1)  # Averages the temporal dimension and obtains shape=(batch, channel_dimension)

        # Check if age is provided and concatenate
        if age is not None:
            # Ensure age is the right shape (B, 1) where B is the batch size
            if age.dim() == 1:
                age = age.view(-1, 1)
            # Concatenate age along the feature dimension
            x = torch.cat((x, age), dim=1)

            x = self._fc_layer_age(x)
            return x

        else:

            # Pass through FC layer and return. No activation function used
            x = self._fc_layer(x)
            return self._dropout(x)


# ------------------
# Functions
# ------------------
def _no_activation_function(x: torch.Tensor) -> torch.Tensor:
    """This can be used as activation function if no activation function is wanted. It is typically more convenient to
    use this function, instead of handling activation functions of type None"""
    return x
