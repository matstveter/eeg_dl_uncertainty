from os.path import split
from typing import Optional, Tuple

import torch
import random
from torch.utils.data import Dataset

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.dataset.OODDataset import BaseDataset


class CauDataGenerator(Dataset):  # type: ignore[type-arg]
    def __init__(self, subjects: Tuple[str, ...], split: str, dataset: CauEEGDataset, use_age: bool, augmentations=None,
                 device: Optional[torch.device] = None, age_noise_prob=0.0, age_noise_level: float = 0.1,
                 clamp_age: bool = False):
        super().__init__()
        self._use_age = use_age
        self.split = split
        self.augmentations = augmentations if augmentations is not None else []

        self.age_noise_prob = age_noise_prob
        self.age_noise_level = age_noise_level

        self.clamp_age = clamp_age

        self.ages = torch.tensor(dataset.load_ages(subjects=subjects), dtype=torch.float32)

        # Check if negative age values are present, this suggests that sklearn standardscaling was used
        if torch.any(self.ages < 0):
            self.clamp_age = False

        x, self._subject_keys = dataset.load_eeg_data(subjects=subjects, split=split)
        self._x = torch.tensor(x, dtype=torch.float32)

        targets = torch.tensor(dataset.load_targets(subjects=subjects, split=split),
                               dtype=torch.float32)

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

    def apply_augmentations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for aug in self.augmentations:
            x = aug.forward(x)
        return x

    def apply_age_noise(self, age_tensor: torch.Tensor) -> torch.Tensor:
        local_random = random.Random()
        ran = local_random.random()
        if ran < self.age_noise_prob:
            noise = torch.normal(mean=0, std=self.age_noise_level, size=age_tensor.shape)
            age_tensor += noise

        return age_tensor

    def __len__(self) -> int:  # type: ignore[no-any-return]
        return self._x.size()[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.augmentations and self.split == 'train':
            eeg_data = self.apply_augmentations(self._x[index])
        else:
            eeg_data = self._x[index]

        if self._use_age:
            # Create and attach age tensor
            age_tensor = self.ages[index].clone().detach().view(1, -1)
            if self.age_noise_prob > 0.0 and self.split == 'train':
                age_tensor = self.apply_age_noise(age_tensor)
                if self.clamp_age:
                    # We only clamp if we use min-max scaling, and if the noise surpasses the bounds
                    age_tensor = torch.clamp(age_tensor, min=0, max=1)

            age_tensor = age_tensor.expand(1, eeg_data.shape[1])
            combined_data = torch.cat((eeg_data, age_tensor), dim=0)

            if self.split == 'test':
                return combined_data, self._y[index], index
            else:
                return combined_data, self._y[index]
        else:
            if self.split == 'test':
                return eeg_data, self._y[index], index
            else:
                return eeg_data, self._y[index]

    def get_subject_keys_from_indices(self, indices):
        return [self._subject_keys[i] for i in indices]


class OODDataGenerator(Dataset):  # type: ignore[type-arg]
    def __init__(self, dataset: BaseDataset, use_age: bool, device: Optional[torch.device] = None):
        super().__init__()
        self._use_age = use_age
        self.ages = torch.tensor(dataset.load_ages(), dtype=torch.float32)
        x, self._subject_keys = dataset.load_eeg_data()
        self._x = torch.tensor(x, dtype=torch.float32)
        targets = torch.tensor(dataset.load_targets(), dtype=torch.float32)

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
        if self._use_age:
            age_tensor = self.ages[index].clone().detach().view(1, -1)
            age_tensor = age_tensor.expand(1, self._x[index].shape[1])
            combined_data = torch.cat((self._x[index], age_tensor), dim=0)
            return combined_data, self._y[index], index
        else:
            return self._x[index], self._y[index], index

    def get_subject_keys_from_indices(self, indices):
        return [self._subject_keys[i] for i in indices]
