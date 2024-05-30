from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class CauDataGenerator(Dataset):  # type: ignore[type-arg]
    def __init__(self, subjects: Tuple[str, ...], split: str, dataset: CauEEGDataset, use_age: bool,
                 device: Optional[torch.device] = None):
        super().__init__()
        self._use_age = use_age
        if split == "train":
            self.ages = torch.tensor(dataset.load_ages(subjects=subjects, add_noise=False), dtype=torch.float32)
        else:
            self.ages = torch.tensor(dataset.load_ages(subjects=subjects), dtype=torch.float32)
        self._x = torch.tensor(dataset.load_eeg_data(subjects=subjects, split=split), dtype=torch.float32)
        targets = torch.tensor(dataset.load_targets(subjects=subjects, split=split), dtype=torch.float32)

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
            return combined_data, self._y[index]
        else:
            return self._x[index], self._y[index]
