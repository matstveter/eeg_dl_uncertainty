import warnings
from typing import Optional, Tuple

import mne.io
import numpy as np
import torch
from scipy.signal import hilbert
from torch.utils.data import Dataset

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset

# todo For channel augmentation, use random channels, but reproducible?
# todo For phase shift, baseline, scalar, gaussian, should we use the same amount of noise for all channels?
# todo We want to use random values for channels and a fixed seed, that should be a boolean parameter
# todo Interpolate, keep two channels always, Cz and one more
# todo Phase shift using circular or hilbert?
# todo Shift the alpha peak, using frequency shifts


class EEGDatashiftGenerator(Dataset):
    def __init__(self, subjects: Tuple[str, ...], dataset: CauEEGDataset, use_age: bool, shift_intensity, shift_type,
                 random_seed=None, device: Optional[torch.device] = None, **kwargs):
        super().__init__()
        self._use_age = use_age
        self._shift_type = shift_type
        self._eeg_info = dataset.eeg_info
        self._use_notch = False

        if self._shift_type == "baseline_drift":
            if "max_drift_amplitude" not in kwargs:
                raise ValueError("max_drift_amplitude must be provided for baseline drift augmentation")
            if "num_sinusoids" not in kwargs:
                raise ValueError("num_sinusoids must be provided for baseline drift augmentation")

            self._max_drift_amplitude = kwargs.pop("max_drift_amplitude")
            self._num_sinusoids = kwargs.pop("num_sinusoids")
        elif self._shift_type in ("scalar_modulation_channel", "scalar_modulation"):
            if "scalar_multi" not in kwargs:
                raise ValueError("scalar_multi must be provided for scalar modulation augmentation")
            self._scalar_multi = kwargs.pop("scalar_multi")
        elif self._shift_type in ("phase_shift_channel", "phase_shift"):
            if "phase_shift" not in kwargs:
                raise ValueError("phase_shift must be provided for phase shift augmentation")
            self._phase_shift = kwargs.pop("phase_shift")
        elif self._shift_type in ("gaussian_channel", "gaussian"):
            if "gaussian_std" not in kwargs:
                raise ValueError("gaussian_std must be provided for gaussian noise augmentation")
            self._gaussian_std = kwargs.pop("gaussian_std")

        # Create a single RNG for the entire class
        if random_seed is None:
            # Otherwise, set a fixed seed for reproducibility
            self.rng = np.random.default_rng(random_seed)

        if not 1.0 >= shift_intensity >= 0:
            raise ValueError("Shift intensity should be between 0.0 and 1.0.")

        self._shift_intensity = shift_intensity
        self.ages = torch.tensor(dataset.load_ages(subjects=subjects, split="test"), dtype=torch.float32)

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
        if self._shift_type in ("gaussian_channel", "phase_shift_channel", "scalar_modulation_channel",
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
        elif self._shift_type == "phase_shift":
            return self._phase_shift_eeg(data=data, phi=self._phase_shift)
        elif self._shift_type == "baseline_drift":
            return self._add_baseline_drift(data)
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

        Parameters
        ----------
        data: np.ndarray
            [n_subject, n_channels, n_timesteps] array with eegs for the subjects in the test set

        Returns
        -------
        altered_array: np.ndarray
            Array with the changes implemented
        """

        augment_order = (1, 3, 5, 7, 9, 11, 13, 15, 17, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
        cur_channel_to_augment = augment_order[0: int(self._shift_intensity * len(augment_order))]

        altered_array = np.zeros_like(data)

        for i, sub in enumerate(data):
            for j, ch in enumerate(sub):
                # If the current channel is in the list of channels that should be reversed
                if j in cur_channel_to_augment:
                    if self._shift_type == "gaussian_channel":
                        raise NotImplementedError("Check todos")
                        noise = self.rng.normal(0, self._gaussian_std, ch.shape)
                        new_arr = ch + noise
                    elif self._shift_type == "phase_shift_channel":
                        raise NotImplementedError("Check todos")
                        # Plot difference after shift
                        new_arr = self._phase_shift_eeg(data=ch, phi=self._phase_shift)
                    elif self._shift_type == "scalar_modulation_channel":
                        raise NotImplementedError("Check todos")
                        new_arr = ch * self._scalar_multi
                    elif self._shift_type in ("theta_bandpass", "alpha_bandpass", "beta_bandpass", "delta_bandpass",
                                              "gamma_bandpass"):
                        eeg_data = np.copy(ch)
                        h_freq, l_freq = self._get_freq()
                        # Ensure data is in 2D format
                        eeg_data_2d = eeg_data[np.newaxis, :]  # Shape becomes (1, 2000)
                        # Create MNE info structure
                        info = mne.create_info(ch_names=['EEG 001'], sfreq=self._eeg_info['sfreq'],
                                               ch_types=['eeg'])
                        # Create Raw object
                        raw = mne.io.RawArray(eeg_data_2d, info, verbose=False)
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
        raise NotImplementedError("Check todos, remember to sample channels, keep Cz and one more")
        # Define the order of channels to augment, excluding the first channel
        # channels_to_keep = [17, 18] Pz and Cz
        augment_order = [1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 16]

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

        # Loop through each subject's data and interpolate bad channels
        for i, sub in enumerate(data):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    raw = mne.io.RawArray(sub, info_object, verbose=False)
                    raw.info['bads'] = channels_to_interpolate
                    raw.interpolate_bads(verbose=False, method="MNE")
                    altered_array[i] = raw.get_data()
            except FutureWarning:
                # This block may not be necessary as the warning is suppressed
                pass

        return altered_array

    def _gaussian(self, data):
        """
        Apply Gaussian noise to the given EEG data.

        This function takes a 2D array of EEG data and adds Gaussian noise to each data point.
        The noise is generated with a mean of 0 and a standard deviation equal to the shift intensity.

        Parameters
        ----------
        data : array_like
            The input EEG data to which the noise will be applied. This should be a 2D array.

        Returns
        -------
        array_like
            The EEG data after the noise has been applied. This is a 2D array of the same size as the input data.

        Notes
        -----
        The function starts by creating a new numpy array, `altered_array`, which has the same shape as the input data.
        This array is initialized with zeros and will be used to store the noisy data.

        The function then loops over the input data. For each subject's data (referred to as `sub` in the code),
        it generates a noise array with the same shape as `sub` using `np.random.normal(0, self._shift_intensity,
        sub. shape)`.
        This noise array is then added to `sub` to create the noisy data, which is stored in the corresponding position
        in `altered_array`.
        """
        altered_array = np.zeros_like(data)

        for i, sub in enumerate(data):
            # todo Is this correct???
            noise = self.rng.normal(0, self._gaussian_std * self._shift_intensity, sub.shape)
            noisy_sub = sub + noise
            altered_array[i] = noisy_sub

            return altered_array

    def _get_freq(self):
        """
        Determine the frequency band based on the shift type.

        This method extracts the frequency band from the shift type, which is expected to be a string
        in the format "<frequency_band>_<other_info>". The frequency band is the first part of the string,
        before the underscore.

        The method supports the following frequency bands: alpha, theta, delta, beta, lbeta, hbeta, and gamma.
        Each frequency band corresponds to a specific range of frequencies, which are returned as a tuple
        of the form (low_frequency, high_frequency).

        If the frequency band is not recognized, the method raises a ValueError.

        Returns
        -------
        tuple
            A tuple of two integers representing the low and high frequencies of the band.

        Raises
        ------
        ValueError
            If the frequency band is not recognized.
        """
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
            return 30, 100
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
        it multiplies the data by the shift intensity and stores the result in the corresponding position
        in `altered_array`.
        """
        altered_array = np.zeros_like(data)

        print(f"Scalar modulation with intensity: {self._shift_intensity}")
        for i, sub in enumerate(data):
            altered_array[i] = sub * self._shift_intensity

        return altered_array

    @staticmethod
    def _phase_shift_eeg(data, phi):
        """
            Apply a phase shift to the given data.

            This function takes a 1D array of data and a phase shift value, phi, in radians.
            It first computes the analytical signal of the data using the Hilbert transform.
            Then, it computes the phase of the analytical signal and adds the phase shift, phi, to it.
            Finally, it computes the real part of the absolute value of the analytical signal multiplied by
            the exponential of the phase data (which includes the phase shift).
            The function returns this modified data.

            Parameters
            ----------
            data : array_like
                The input data to which the phase shift will be applied. This should be a 1D array.
            phi : float
                The phase shift to apply, in radians.

            Returns
            -------
            array_like
                The data after the phase shift has been applied. This is a 1D array of the same size as the input data.
        """
        # Compute the analytical signal of the data using the Hilbert transform
        analytical_signal = hilbert(data)

        # Compute the phase of the analytical signal
        phase_data = np.angle(analytical_signal)

        # Add the phase shift to the phase data
        phase_data += phi

        # Compute the real part of the absolute value of the analytical signal multiplied by the exponential of the
        # phase data
        modified_data = np.real(np.abs(analytical_signal) * np.exp(1j * phase_data))

        return modified_data

    @staticmethod
    def circular_shift_eeg(data: np.ndarray, shift: int) -> np.ndarray:
        """
        Applies a circular shift to EEG data along the time axis.

        Args:
            data (np.ndarray): EEG data. Can be 2D (n_channels x n_times) or
                               3D (n_samples x n_channels x n_times).
            shift (int): Number of samples to shift.

        Returns:
            np.ndarray: Circularly shifted EEG data.
        """
        # If data is 2D, shift along axis=1 (time axis)
        if data.ndim == 2:
            return np.roll(data, shift, axis=1)
        # If data is 3D, shift along axis=2 (time axis)
        elif data.ndim == 3:
            return np.roll(data, shift, axis=2)
        else:
            raise ValueError("Data must be 2D or 3D.")

    def _add_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """
        Adds a combination of low-frequency sinusoids with random frequencies, phases,
        and weights to simulate realistic baseline drifts. A fixed random seed ensures
        reproducible results (useful for consistent model comparisons).

        Returns:
            np.ndarray: Copy of `data` with added baseline drifts (same shape).
        """

        data_drifted = data.copy()
        n_samples, n_channels, n_times = data_drifted.shape

        # Time axis over which we'll define our sinusoidal waves
        time_axis = np.linspace(0, 2 * np.pi, n_times)

        # Overall amplitude of the drift
        drift_amplitude = self._max_drift_amplitude * self._shift_intensity

        # Generate multiple sinusoids
        sinusoids = []
        for _ in range(self._num_sinusoids):
            freq = self.rng.uniform(0.3, 1.0)  # random low frequency
            phase = self.rng.uniform(0, 2 * np.pi)  # random starting phase
            wave = np.sin(freq * time_axis + phase)
            sinusoids.append(wave)

        # Add drift to each channel
        for i in range(n_samples):
            for ch in range(n_channels):
                combined = np.zeros_like(time_axis)
                for wave in sinusoids:
                    weight = self.rng.uniform(0.5, 1.0)
                    combined += weight * wave

                data_drifted[i, ch, :] += drift_amplitude * combined

        return data_drifted

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
