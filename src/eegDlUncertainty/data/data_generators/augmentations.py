from typing import List, Optional
from braindecode.augmentation import GaussianNoise, TimeReverse, SignFlip, FTSurrogate, ChannelsShuffle, \
    ChannelsDropout, SmoothTimeMask, BandstopFilter, Transform


def get_augmentations(aug_names: List[str], probability: float, random_state: Optional[int] = None) -> List[Transform]:
    """
    Constructs a list of augmentation objects based on the specified augmentation names. Each augmentation
    is initialized with a given probability and an optional random state for reproducibility. Supported augmentations
    include Gaussian noise addition, time series reversal, signal flipping, Fourier transform surrogates, channel shuffling,
    channel dropout, smooth time masking, and band-stop filtering. This functionality is inspired by and sampled from
    Braindecode, a library dedicated to deep learning with EEG brainwave data.

    Parameters
    ----------
    aug_names : List[str]
        A list of strings specifying the names of the augmentations to be created. Accepted names include various
        case-insensitive versions of 'GaussianNoise', 'TimeReverse', 'SignFlip', 'FTSurrogate', 'ChannelsShuffle',
        'ChannelsDropout', 'SmoothTimeMask', and 'BandstopFilter'.
    probability : float
        The probability with which each augmentation should be applied to a given sample during training.
    random_state : Optional[int], default=None
        An optional integer seed for random number generators to ensure reproducibility of the augmentations.

    Returns
    -------
    List
        A list containing the initialized augmentation objects corresponding to the specified augmentation names.

    Raises
    ------
    KeyError
        If an unrecognized augmentation name is provided in `aug_names`.

    References
    ----------
    For more details on the augmentations and their parameters, refer to the Braindecode documentation:
    https://braindecode.org/0.6/api.html#augmentation

    """
    aug_list = []
    for name in aug_names:
        if name in ('gaussiannoise', 'GaussianNoise'):
            aug_list.append(GaussianNoise(probability=probability, std=0.001, random_state=random_state))
        elif name in ('timereverse', 'TimeReverse'):
            aug_list.append(TimeReverse(probability=probability, random_state=random_state))
        elif name in ('signflip', 'SignFlip'):
            aug_list.append(SignFlip(probability=probability, random_state=random_state))
        elif name in ('ftsurrogate', 'FTSurrogate'):
            aug_list.append(FTSurrogate(probability=probability, random_state=random_state))
        elif name in ('channelsshuffle', 'ChannelsShuffle'):
            aug_list.append(ChannelsShuffle(probability=probability, p_shuffle=0.2, random_state=random_state))
        elif name in ('channelsdropout', 'ChannelsDropout'):
            aug_list.append(ChannelsDropout(probability=probability, p_drop=0.2, random_state=random_state))
        elif name in ('smoothtimemask', 'SmoothTimeMask'):
            aug_list.append(SmoothTimeMask(probability=probability, mask_len_samples=100, random_state=random_state))
        elif name in ('bandstopfilter', 'BandstopFilter'):
            aug_list.append(BandstopFilter(probability=probability, bandwidth=10, sfreq=120, max_freq=40,
                                           random_state=random_state))
        else:
            raise KeyError(f"Unrecognized augmentation technique: {name}")
    return aug_list
