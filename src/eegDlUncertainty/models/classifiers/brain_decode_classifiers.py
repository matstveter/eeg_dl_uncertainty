from typing import Tuple, Union

from braindecode.models import EEGNetv4

from eegDlUncertainty.models.base_classifier import BaseClassifier


class EEGNetv4MTSC(EEGNetv4, BaseClassifier):
    def __init__(self, **kwargs):

        in_channels = kwargs.pop("in_channels")
        num_classes = kwargs.pop("num_classes")

        time_steps = kwargs.get("time_steps")
        final_conv_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        dropout_rate: float = kwargs.get("dropout_rate", 0.25)
        pool_mode: str = kwargs.get("pool_mode", "mean")
        f1: int = kwargs.get("f1", 8)
        d: int = kwargs.get("d", 2)
        f2: int = kwargs.get("f2", 16)
        kernel_length: int = kwargs.get("kernel_length", 64)
        third_kernel_size: Tuple[int, int] = kwargs.get("third_kernel_size", (8, 4))

        # Initialise by calling super class
        super().__init__(in_chans=in_channels, n_classes=num_classes, input_window_samples=time_steps,
                         final_conv_length=final_conv_length, pool_mode=pool_mode, F1=f1, D=d, F2=f2,
                         kernel_length=kernel_length, third_kernel_size=third_kernel_size, drop_prob=dropout_rate)
