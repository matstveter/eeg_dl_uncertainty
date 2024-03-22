from typing import Callable, Tuple, Union
from torch.nn.functional import elu
from braindecode.models.functions import identity

import torch.nn
from braindecode.models import (EEGNetv4, EEGNetv1, EEGITNet, Deep4Net)

from eegDlUncertainty.models.base_classifier import BaseClassifier


class EEGITNetMTSC(EEGITNet, BaseClassifier):
    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")
        time_steps: int = kwargs.pop("time_steps")

        # -----------------------
        # Optional kwargs
        # -----------------------
        dropout_rate: float = kwargs.get("dropout_rate", 0.4)

        # Initialise by calling super class
        super().__init__(n_outputs=num_classes, n_chans=in_channels, n_times=time_steps,
                         drop_prob=dropout_rate, add_log_softmax=False)


class Deep4NetMTSC(Deep4Net, BaseClassifier):

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time_steps or final_conv_length must be set
        time_steps: int = kwargs.get("time_steps")  # type: ignore[assigment]
        final_conv_length: Union[int, str] = kwargs.get("final_conv_length", "auto")

        n_filters_time: int = kwargs.get("n_filters_time", 25)
        n_filters_spat: int = kwargs.get("n_filters_spat", 25)
        filter_time_length: int = kwargs.get("filter_time_length", 10)
        pool_time_length: int = kwargs.get("pool_time_length", 3)
        pool_time_stride: int = kwargs.get("pool_time_stride", 3)
        n_filters_2: int = kwargs.get("n_filters_2", 50)
        filter_length_2: int = kwargs.get("filter_length_2", 10)
        n_filters_3: int = kwargs.get("n_filters_3", 100)
        filter_length_3: int = kwargs.get("filter_length_3", 10)
        n_filters_4: int = kwargs.get("n_filters_4", 200)
        filter_length_4: int = kwargs.get("filter_length_4", 10)
        first_conv_nonlin: Callable = kwargs.get("first_conv_nonlin", elu)  # type: ignore[type-arg]
        first_pool_mode: str = kwargs.get("first_pool_mode", "max")
        first_pool_nonlin: Callable = kwargs.get("first_pool_nonlin", identity)  # type: ignore[type-arg]
        later_conv_nonlin: Callable = kwargs.get("later_conv_nonlin", elu)  # type: ignore[type-arg]
        later_pool_mode: str = kwargs.get("later_pool_mode", "max")  # "max" or "mean"
        later_pool_nonlin: Callable = kwargs.get("later_pool_nonlin", identity)  # type: ignore[type-arg]
        dropout_rate: float = kwargs.get("dropout_rate", 0.5)
        split_first_layer: bool = kwargs.get("split_first_layer", True)
        batch_norm: bool = kwargs.get("batch_norm", True)
        batch_norm_alpha: float = kwargs.get("batch_norm_alpha", 0.1)
        stride_before_pool: bool = kwargs.get("stride_before_pool", False)

        # Initialise by calling super class
        super().__init__(n_chans=in_channels, n_outputs=num_classes, drop_prob=dropout_rate,
                         n_times=time_steps, final_conv_length=final_conv_length,
                         n_filters_time=n_filters_time, n_filters_spat=n_filters_spat,
                         filter_time_length=filter_time_length, pool_time_length=pool_time_length,
                         pool_time_stride=pool_time_stride, n_filters_2=n_filters_2, filter_length_2=filter_length_2,
                         n_filters_3=n_filters_3, filter_length_3=filter_length_3, n_filters_4=n_filters_4,
                         filter_length_4=filter_length_4, first_conv_nonlin=first_conv_nonlin,
                         first_pool_mode=first_pool_mode, first_pool_nonlin=first_pool_nonlin,
                         later_conv_nonlin=later_conv_nonlin, later_pool_mode=later_pool_mode,
                         later_pool_nonlin=later_pool_nonlin, split_first_layer=split_first_layer,
                         batch_norm=batch_norm, batch_norm_alpha=batch_norm_alpha,
                         stride_before_pool=stride_before_pool, add_log_softmax=False)


class EEGNetv1MTSC(EEGNetv1, BaseClassifier):

    def __init__(self, **kwargs):
        """
        Initialise
        Args:
            **kwargs: See below
        Examples:
            Traceback (most recent call last):
            ...
            AssertionError
        """
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time steps or final conv length must be specified
        time_steps: int = kwargs.get("time_steps")
        final_conv_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        dropout_rate: float = kwargs.get("dropout_rate", 0.25)
        pool_mode: str = kwargs.get("pool_mode", "max")
        second_kernel_size: Tuple[int, int] = kwargs.get("second_kernel_size", (2, 32))
        third_kernel_size: Tuple[int, int] = kwargs.get("third_kernel_size", (8, 4))

        # Initialise by calling super class
        super().__init__(n_chans=in_channels, n_outputs=num_classes, n_times=time_steps,
                         final_conv_length=final_conv_length, pool_mode=pool_mode,
                         second_kernel_size=second_kernel_size, third_kernel_size=third_kernel_size,
                         drop_prob=dropout_rate, add_log_softmax=False)


class EEGNetv4MTSC(EEGNetv4, BaseClassifier):

    def __init__(self, **kwargs):
        # -----------------------
        # Required kwargs
        # -----------------------
        in_channels: int = kwargs.pop("in_channels")
        num_classes: int = kwargs.pop("num_classes")

        # -----------------------
        # Optional kwargs
        # -----------------------
        # Either time steps or final conv length must be specified
        time_steps: int = kwargs.get("time_steps")
        final_conv_length: Union[str, int] = kwargs.get("final_conv_length", "auto")

        dropout_rate: float = kwargs.get("dropout_rate", 0.25)
        pool_mode: str = kwargs.get("pool_mode", "mean")
        f1: int = kwargs.get("f1", 8)
        d: int = kwargs.get("d", 2)
        f2: int = kwargs.get("f2", 16)  # BrainDecode implementation suggests d*f1 instead?
        kernel_length: int = kwargs.get("kernel_length", 64)
        third_kernel_size: Tuple[int, int] = kwargs.get("third_kernel_size", (8, 4))

        # Initialise by calling super class
        super().__init__(n_chans=in_channels, n_outputs=num_classes, n_times=time_steps,
                         final_conv_length=final_conv_length, pool_mode=pool_mode, F1=f1, D=d, F2=f2,
                         kernel_length=kernel_length, third_kernel_size=third_kernel_size, drop_prob=dropout_rate)

