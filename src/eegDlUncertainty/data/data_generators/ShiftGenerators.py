from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class EEGDatashift(Dataset):
    def __init__(self, subjects: Tuple[str, ...], dataset: CauEEGDataset, use_age: bool, shift_intensity, shift_type,
                 device: Optional[torch.device] = None):
        super().__init__()
        self._use_age = use_age
        self._shift_type = shift_type

        if not 1.0 >= shift_intensity >= 0:
            raise ValueError("Shift intensity should be between 0.0 and 1.0.")

        self._shift_intensity = shift_intensity
        self.ages = torch.tensor(dataset.load_ages(subjects=subjects), dtype=torch.float32)

        inputs = dataset.load_eeg_data(subjects=subjects, split="test")
        targets = dataset.load_targets(subjects=subjects, split="test")

        if self._shift_intensity > 0.0:
            shifted_eeg = self.apply_shift(data=inputs, targets=targets)
        else:
            shifted_eeg = inputs

        self._x = torch.tensor(shifted_eeg, dtype=torch.float32)
        self._y = torch.tensor(targets, dtype=torch.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def apply_shift(self, data, targets):
        if self._shift_type == "class_combination":
            return self._combine_eeg(data=data, targets=targets)

    def _combine_eeg(self, data, targets):
        """
        Combine EEG data from different classes by mixing specified channels.

        This method mixes EEG data from three classes (normal, MCI, and dementia)
        based on specified channels and returns the combined mixed EEG data.

        This mixes normal with dementia, dementia with normal and mci with dementia.

        Parameters
        ----------
        data : array_like
            The EEG data array with shape (samples, channels).
        targets : array_like
            The target labels corresponding to the EEG data. It should be a one-hot encoded
            array with shape (samples, num_classes).

        Returns
        -------
        array_like
            A new EEG data array with mixed channels from different classes.

        Raises
        ------
        ValueError
            If there are fewer dementia class samples than normal or MCI class samples.

        Notes
        -----
        The method assumes there are three classes: normal, MCI, and dementia, which are
        represented by 0, 1, and 2 in the `targets` array, respectively.

        """
        mix_order = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        channels_to_mix = mix_order[0: int(self._shift_intensity * 10)]

        # Extract the classes from the targets, in order to divide the data into seperate classes
        classes = np.argmax(targets, axis=1)
        normal_indices = np.where(classes == 0)[0]
        mci_indices = np.where(classes == 1)[0]
        dementia_indices = np.where(classes == 2)[0]

        # Ensure there are enough dementia indices to match normal and MCI lengths
        if len(dementia_indices) < len(normal_indices):
            dementia_normal_indices = np.concatenate([dementia_indices,
                                                      dementia_indices[0: len(normal_indices) - len(dementia_indices)]])
            normal_dementia_indices = normal_indices[0: len(dementia_indices)]
        else:
            raise ValueError("Normal class less than dementia class? That is not correct....for caueeg")

        if len(dementia_indices) < len(mci_indices):
            dementia_mci_indices = np.concatenate([dementia_indices,
                                                   dementia_indices[0: len(mci_indices) - len(dementia_indices)]])
        else:
            raise ValueError("MCI class less than dementia class? That is not correct....for caueeg..")

        new_normal_eeg = self._mix_eeg(indices_current=normal_indices,
                                       indices_class_to_mix=dementia_normal_indices,
                                       eeg=data, channels_to_mix=channels_to_mix)
        new_mci_eeg = self._mix_eeg(indices_current=mci_indices,
                                    indices_class_to_mix=dementia_mci_indices,
                                    eeg=data, channels_to_mix=channels_to_mix)
        new_dementia_eeg = self._mix_eeg(indices_current=dementia_indices,
                                         indices_class_to_mix=normal_dementia_indices,
                                         eeg=data, channels_to_mix=channels_to_mix)

        all_mixed_eeg = np.zeros_like(data)
        self._place_eeg(indices=normal_indices, mixed_eeg=new_normal_eeg, final_mix_array=all_mixed_eeg)
        self._place_eeg(indices=mci_indices, mixed_eeg=new_mci_eeg, final_mix_array=all_mixed_eeg)
        self._place_eeg(indices=dementia_indices, mixed_eeg=new_dementia_eeg, final_mix_array=all_mixed_eeg)
        return all_mixed_eeg

    @staticmethod
    def _place_eeg(indices, mixed_eeg, final_mix_array):
        """
         Place mixed EEG data into the final array at specified indices.

         This function updates the `final_mix_array` by placing the `mixed_eeg`
         data at the positions specified by `indices`.

         Parameters
         ----------
         indices : array_like
             Indices where the mixed EEG data should be placed.
         mixed_eeg : array_like
             The mixed EEG data array with shape (samples, channels).
         final_mix_array : array_like
             The final EEG data array to be updated with mixed data.

         Returns
         -------
         None
         """
        for ind in indices:
            final_mix_array[ind] = mixed_eeg[ind]

    @staticmethod
    def _mix_eeg(indices_current, indices_class_to_mix, eeg, channels_to_mix):
        """
        Mix specified EEG channels from one set of indices with another.

        This function takes EEG data and mixes specified channels from one set
        of indices with corresponding channels from another set of indices.

        Parameters
        ----------
        indices_current : array_like
            Indices of the current class.
        indices_class_to_mix : array_like
            Indices of the class to mix with.
        eeg : array_like
            The EEG data array with shape (samples, channels).
        channels_to_mix : array_like
            The channels to mix between the indices.

        Returns
        -------
        array_like
            A new EEG data array with mixed channels.

        Notes
        -----
        This function assumes that `indices_current` and `indices_class_to_mix`
        are of the same length and that `channels_to_mix` contains valid channel indices.
        """
        eeg_cp = np.copy(eeg)

        # Define an array on where to put the results
        finished = np.zeros_like(eeg_cp)

        for ind_cur, ind_mix in zip(indices_current, indices_class_to_mix):
            eeg_cur = eeg_cp[ind_cur]
            eeg_mix = eeg_cp[ind_mix]

            for i in range(eeg_cur.shape[0]):
                if i in channels_to_mix:
                    eeg_cur[i] = eeg_mix[i]

            finished[ind_cur] = eeg_cur
        return finished

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
