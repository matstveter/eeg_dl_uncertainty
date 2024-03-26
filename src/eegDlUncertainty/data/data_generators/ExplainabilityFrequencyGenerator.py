from typing import Optional, Tuple

import mne.io
import numpy as np
from torch.utils.data import Dataset
import torch

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class ExplainabilityFrequencyGenerator(Dataset):  # type: ignore[type-arg]
    def __init__(self, subjects: Tuple[str, ...], dataset: CauEEGDataset, frequency_band: str,
                 device: Optional[torch.device] = None):
        super().__init__()
        if frequency_band in ("delta", "theta", "alpha", "low_beta", "high_beta", "gamma"):
            self.frequency_band = frequency_band
            frequency_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 12),
                "low_beta": (12, 20),
                "high_beta": (20, 30),
                "gamma": (30, 40)
            }
            freqs = frequency_bands[self.frequency_band]
        else:
            raise ValueError(f"Frequency band : {frequency_band} not recognized!")

        raw_data = dataset.load_eeg_data(subjects=subjects)
        processed_data = self.apply_bandpass_exclusion(data=raw_data, dataset=dataset, freq_band=freqs)

        self._x = torch.tensor(processed_data, dtype=torch.float32)
        targets = torch.tensor(dataset.load_targets(subjects=subjects), dtype=torch.float32)

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        self._y = targets
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    # ---------------
    # Properties
    # ---------------
    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    def __len__(self) -> int:  # type: ignore[no-any-return]
        return self._x.size()[0]

    def __getitem__(self, index):
        return self._x[index], self._y[index]

    @staticmethod
    def apply_bandpass_exclusion(data, dataset, freq_band: list):
        """
        Applies a band exclusion filter to a batch of EEG data using MNE.

        Parameters:
        - data: Numpy array of shape (subjects, channels, samples), the EEG data.
        - dataset: Instance of the dataset class providing EEG info (like channel names and sampling frequency).
        - freq_band: List with two elements [l_freq, h_freq] defining the band to be excluded.

        Returns:
        - processed_data: Numpy array of the same shape as `data`, with the specified frequency band excluded.
        """
        processed_data = np.zeros_like(data)

        for i, s in enumerate(data):
            raw = mne.io.RawArray(s, info=dataset.get_eeg_info())
            raw.filter(l_freq=freq_band[1], h_freq=freq_band[0])
            processed_data[i] = raw.get_data()

        return processed_data
