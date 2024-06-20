from typing import Optional, Tuple

import mne.io
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class EEGDatashiftGenerator(Dataset):
    def __init__(self, subjects: Tuple[str, ...], dataset: CauEEGDataset, use_age: bool, shift_intensity, shift_type,
                 device: Optional[torch.device] = None, scalar_multi: Optional[float] = 0.5):
        super().__init__()
        self._use_age = use_age
        self._shift_type = shift_type
        self._eeg_info = dataset.eeg_info
        self._use_notch = True

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

        self._scalar_multi = scalar_multi
        self._x = torch.tensor(shifted_eeg, dtype=torch.float32)
        self._y = torch.tensor(targets, dtype=torch.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def apply_shift(self, data, targets):
        if self._shift_type == "class_combination_temporal":
            return self._combine_eeg(data=data, targets=targets, spatial=False)
        elif self._shift_type in ("gaussian_channel", "phase_shift_channel", "scalar_modulation_channel",
                                  "alpha_bandpass", "beta_bandpass",
                                  "theta_bandpass", "delta_bandpass", "gamma_bandpass", "hbeta_bandpass",
                                  "lbeta_bandpass"):
            return self._channel_augmentations(data)
        elif self._shift_type == "interpolate":
            return self._interpolate_augment(data=data)
        elif self._shift_type == "scalar_modulation":
            return self._scalar_modulation(data=data)
        elif self._shift_type == "gaussian":
            return self._gaussian(data=data)
        else:
            raise KeyError("Shift type not recognized")

    def _combine_eeg(self, data, targets, spatial=True):
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
                                       eeg=data, channels_to_mix=channels_to_mix, spatial=spatial)
        new_mci_eeg = self._mix_eeg(indices_current=mci_indices,
                                    indices_class_to_mix=dementia_mci_indices,
                                    eeg=data, channels_to_mix=channels_to_mix, spatial=spatial)
        new_dementia_eeg = self._mix_eeg(indices_current=dementia_indices,
                                         indices_class_to_mix=normal_dementia_indices,
                                         eeg=data, channels_to_mix=channels_to_mix, spatial=spatial)

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

    def _mix_eeg(self, indices_current, indices_class_to_mix, eeg, channels_to_mix, spatial):
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

            if spatial:
                for i in range(eeg_cur.shape[0]):
                    if i in channels_to_mix:
                        eeg_cur[i] = eeg_mix[i]
            else:
                # Maximum augmentation is set to 0.5 of the original signal
                keep_time_steps = int((eeg_cur.shape[1] * (1 - self._shift_intensity * 0.5)))

                eeg_part_cur = eeg_cur[:, 0: keep_time_steps]
                eeg_part_mix = eeg_mix[:, keep_time_steps:]
                eeg_cur = np.concatenate((eeg_part_cur, eeg_part_mix), axis=1)

            finished[ind_cur] = eeg_cur
        return finished

    def _channel_augmentations(self, data):
        """ THis channel performs shifts on channels.

        The augmentation order is specified as every other channel, and
        then the channels that were skipped. It uses the shift_intensity to sub-select which channels that should be
        augmented. It loops through the data array and then if the channel of a subject is in the sub-selected
        channels:

        Timereverse: Flip the signal so that the first time step is the last

        Shift: Multiply the signal with -1, effectively flipping the sign of the signal

        Parameters
        ----------
        data: np.ndarray
            [n_subject, n_channels, n_timesteps] array with eegs for the subjects in the test set

        Returns
        -------
        altered_array: np.ndarray
            Array with the changes implemented
        """

        augment_order = [1, 3, 5, 7, 9, 11, 13, 15, 17, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        cur_channel_to_augment = augment_order[0: int(self._shift_intensity * len(augment_order))]

        altered_array = np.zeros_like(data)

        for i, sub in enumerate(data):
            for j, ch in enumerate(sub):
                # If the current channel is in the list of channels that should be reversed
                if j in cur_channel_to_augment:
                    if self._shift_type == "gaussian_channel":
                        noise = np.random.normal(0, 0.1, ch.shape)
                        new_arr = ch + noise
                    elif self._shift_type == "phase_shift_channel":
                        # Plot difference after shift
                        plt.plot(ch)
                        plt.plot(self._phase_shift_data(ch, np.pi/2))
                        plt.show()
                        new_arr = self._phase_shift_data(ch, np.pi/2)
                    elif self._shift_type == "scalar_modulation_channel":
                        new_arr = ch * self._scalar_multi
                    elif self._shift_type in ("theta_bandpass", "alpha_bandpass", "beta_bandpass", "delta_bandpass",
                                              "hbeta_bandpass", "lbeta_bandpass", "gamma_bandpass"):
                        eeg_data = np.copy(ch)
                        h_freq, l_freq = self._get_freq()
                        # Ensure data is in 2D format
                        eeg_data_2d = eeg_data[np.newaxis, :]  # Shape becomes (1, 2000)
                        # Create MNE info structure
                        info = mne.create_info(ch_names=['EEG 001'], sfreq=self._eeg_info['sfreq'],
                                               ch_types=['eeg'])
                        # Create Raw object
                        raw = mne.io.RawArray(eeg_data_2d, info)
                        # High-pass filter to remove frequencies below 12 Hz
                        raw_high_pass = raw.copy().filter(l_freq=l_freq, h_freq=None, verbose=False)

                        # Low-pass filter to remove frequencies above 8 Hz
                        raw_low_pass = raw.copy().filter(l_freq=None, h_freq=h_freq, verbose=False)

                        # Combine the filtered signals by adding the data
                        combined_data = raw_high_pass.get_data() + raw_low_pass.get_data()
                        new_arr = combined_data
                    else:
                        raise ValueError("Unrecognized shift type")
                else:
                    new_arr = ch
                altered_array[i][j] = new_arr

        return altered_array

    def _interpolate_augment(self, data):
        """
        Interpolate bad channels in EEG data and augment the dataset.

        Parameters
        ----------
        data : array-like
            The EEG data to be augmented. Each element in the array represents
            the EEG recording for a subject.

        Returns
        -------
        altered_array : ndarray
            The augmented EEG data with bad channels interpolated.
        """
        # Define the order of channels to augment, excluding the first channel
        augment_order = [1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        # Determine the channels to augment based on the shift intensity
        cur_channel_to_augment = augment_order[0: int(self._shift_intensity * len(augment_order))]

        # Extract EEG channel information
        info_object = self._eeg_info
        montage = mne.channels.make_standard_montage('standard_1020')
        info_object.set_montage(montage)
        channel_names = info_object['ch_names']

        # Select the channel names to interpolate based on the indices
        channels_to_interpolate = [channel_names[i] for i in cur_channel_to_augment]

        # Initialize an array to store the altered EEG data
        altered_array = np.zeros_like(data)

        print(channels_to_interpolate)
        # Loop through each subject's data and interpolate bad channels
        for i, sub in enumerate(data):
            raw = mne.io.RawArray(sub, info_object, verbose=False)
            raw.info['bads'] = channels_to_interpolate
            raw.interpolate_bads(verbose=False, method="MNE")
            altered_array[i] = raw.get_data()

        return altered_array

    def _gaussian(self, data):
        altered_array = np.zeros_like(data)

        for i, sub in enumerate(data):
            noise = np.random.normal(0, self._shift_intensity, sub.shape)
            noisy_sub = sub + noise
            altered_array[i] = noisy_sub

        return altered_array

    def _get_freq(self):
        freq = self._shift_type.split("_")[0]

        if freq == "alpha":
            return 8, 12
        elif freq == "theta":
            return 4, 8
        elif freq == "delta":
            return 1, 4
        elif freq == "beta":
            return 12, 30
        elif freq == "lbeta":
            return 12, 20
        elif freq == "hbeta":
            return 20, 30
        elif freq == "gamma":
            return 30, 50
        else:
            raise ValueError(f"Frequency band: {freq} is not recognized")

    def _scalar_modulation(self, data):
        """
        Perform scalar modulation on the given EEG data.

        This function takes a 2D array of EEG data and multiplies each data point by a shift intensity factor.
        The shift intensity is a class attribute representing the factor by which the data will be multiplied.

        Parameters
        ----------
        data : array_like
            The input EEG data to be modulated. This should be a 1D array.

        Returns
        -------
        array_like
            The modulated EEG data. This is a 1D array of the same size as the input data.

        Notes
        -----
        The function starts by creating a new numpy array, `altered_array`, which has the same shape as the input data.
        This array is initialized with zeros and will be used to store the modulated data.

        The function then loops over the input data. For each subject's data (referred to as `sub` in the code),
        it multiplies the data by the shift intensity and stores the result in the corresponding position in `altered_array`.
        """
        altered_array = np.zeros_like(data)

        print(f"Scalar modulation with intensity: {self._shift_intensity}")
        for i, sub in enumerate(data):
            altered_array[i] = sub * self._shift_intensity

        return altered_array

    @staticmethod
    def _phase_shift_data(data, phi):
        """
        Perform a phase shift on the given data.

        This function takes a 1D array of data and a phase shift value in radians,
        performs a Fourier Transform to move to the frequency domain, applies the phase shift,
        and then performs an Inverse Fourier Transform to move back to the time domain.

        Parameters
        ----------
        data : array_like
            The input data to be phase shifted. This should be a 1D array.
        phi : float
            The phase shift to apply, in radians.

        Returns
        -------
        array_like
            The phase-shifted data. This is a 1D array of the same size as the input data.

        Notes
        -----
        The phase shift is performed by multiplying the Fourier Transform of the data by `e^(i*phi)`.
        The result is then transformed back to the time domain using the Inverse Fourier Transform.
        Only the real part of the result is returned, as the imaginary part should be zero (or close to zero
        due to numerical precision issues) after the Inverse Fourier Transform.
        """
        # Fourier Transform
        transformed = np.fft.fft(data)

        # Phase Shift
        shifted = transformed * np.exp(1j * phi)

        # Inverse Fourier Transform
        result = np.fft.ifft(shifted)

        return np.real(result)  # return only the real part

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
