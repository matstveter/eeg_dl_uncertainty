from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class CauDataGenerator(Dataset):  # type: ignore[type-arg]
    def __init__(self, subjects: Tuple[str, ...], split: str, dataset: CauEEGDataset,
                 device: Optional[torch.device] = None):
        super().__init__()
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
        return self._x[index], self._y[index]
