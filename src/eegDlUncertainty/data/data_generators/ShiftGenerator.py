import warnings
from typing import Tuple

import mne.io
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from torch.utils.data import Dataset

from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


class EEGDatashiftGenerator(Dataset):
    def __init__(self, subjects: Tuple[str, ...], dataset: CauEEGDataset, use_age: bool, shift_intensity, shift_type,
                 use_same_shift, random_seed=None, device=None, plot_difference=False, **kwargs):
        super().__init__()
        # Dataset information
        self._shift_type = shift_type
        self._eeg_info = dataset.eeg_info
        self._use_notch = False

        self._shift_intensity = shift_intensity
        self._shift_type = shift_type
        self._same_shift = use_same_shift
        self._random_seed = random_seed
        self._plot_difference = plot_difference
        # Set random generator
        self.rng = np.random.default_rng(seed=random_seed)

        self.ages = torch.tensor(dataset.load_ages(subjects=subjects), dtype=torch.float32)

        if not use_age:
            mean_age = dataset.get_mean_age()
            # Set all ages to the mean age
            self.ages = torch.ones_like(self.ages) * mean_age

        inputs, self._subject_keys = dataset.load_eeg_data(subjects=subjects, split="test")
        targets = dataset.load_targets(subjects=subjects, split="test")

        if shift_type == "phase_shift":
            if "phase_shift" not in kwargs:
                raise ValueError("Phase shift requires the phase shift value")
            self._phase_shift = kwargs.get("phase_shift")
        elif shift_type == "circular_shift":
            if "circular_shift" not in kwargs:
                raise ValueError("Circular shift requires the shift value")
            self._circular_shift = kwargs.get("circular_shift")
        elif shift_type == "amplitude_change":
            if "scalar_multi" not in kwargs:
                raise ValueError("Amplitude change requires the scalar multiplier")
            self._scalar_multi = kwargs.get("scalar_multi")
        elif shift_type == "gaussian":
            if "gaussian_std" not in kwargs:
                raise ValueError("Gaussian shift requires the standard deviation of the noise")
            self._gaussian_std = kwargs.get("gaussian_std")
            # If we want to use the same shift for all channels
            if self._same_shift:
                self._gaussian_std = self.rng.normal(self._gaussian_std, self._gaussian_std * 0.75, 6000)
        elif shift_type == "baseline_drift":
            if "max_drift_amplitude" not in kwargs:
                raise ValueError("Baseline drift requires the maximum drift amplitude")
            self._max_drift_amplitude = kwargs.get("max_drift_amplitude")
            if "num_sinusoids" not in kwargs:
                raise ValueError("Baseline drift requires the number of sinusoids")
            self._num_sinusoids = kwargs.get("num_sinusoids")

            if self._same_shift:
                self._drift = self._calculate_drift_signal(n_times=6000)
        elif shift_type == "timewarp":
            if "max_warp_ratio" not in kwargs:
                raise ValueError("Timewarp requires the maximum warp ratio")
            self._max_warp_ratio = kwargs.get("max_warp_ratio")

            if self._same_shift:
                self._warp_signal = self._calculate_warp_signal(n_times=6000)
        elif shift_type == "alpha_peak_shift":
            if "peak_shift" not in kwargs:
                raise ValueError("Alpha peak shift requires the peak shift value")

            self._peak_shift = kwargs.get("peak_shift")

        if self._shift_intensity > 0.0:
            shifted_eeg = self.apply_shift(data=inputs)
        else:
            shifted_eeg = inputs

        self._x = torch.tensor(shifted_eeg, dtype=torch.float32)
        self._y = torch.tensor(targets, dtype=torch.float32)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def apply_shift(self, data):
        if self._shift_type == "interpolate":
            return self._interpolate_augment(data=data)
        else:
            return self._channel_augmentations(data)

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

        augment_order = np.arange(19)

        # Sort the channels, randomly but reproducible
        self.rng.shuffle(augment_order)
        cur_channel_to_augment = augment_order[0: int(self._shift_intensity * len(augment_order))]

        if self._shift_type == "rotate_channels":
            return self._channel_randomizing(data=data, channels_to_rotate=cur_channel_to_augment)

        altered_array = np.zeros_like(data)

        for i, sub in enumerate(data):
            for j, ch in enumerate(sub):
                # If the current channel is in the list of channels that should be reversed
                if j in cur_channel_to_augment:
                    if self._shift_type == "gaussian":
                        if self._same_shift:
                            noise = self._gaussian_std
                        else:
                            # Sample a random noise value from self._gaussian_std
                            noise = self.rng.normal(self._gaussian_std, self._gaussian_std * 0.75, 6000)

                        new_arr = ch + noise
                    elif self._shift_type == "phase_shift":
                        # Either we use the same shifts for all channels or we are sampling it
                        if self._same_shift:
                            channel_phi = self._phase_shift
                        else:
                            # Sample a random phase shift value from self._phase_shift
                            abs_shift = self.rng.uniform(self._phase_shift * 0.75, self._phase_shift)
                            channel_phi = abs_shift * self.rng.choice([-1, 1])
                        # Plot difference after shift
                        new_arr = self._phase_shift_eeg(data=ch, phi=channel_phi)

                    elif self._shift_type == "circular_shift":
                        # Either we use the same shifts for all channels or we are sampling it
                        if self._same_shift:
                            channel_shift = self._circular_shift
                        else:
                            abs_shift = self.rng.integers(int(self._circular_shift * 0.75), self._circular_shift)
                            channel_shift = abs_shift * self.rng.choice([-1, 1])

                        new_arr = self.circular_shift_eeg(data=ch, shift=channel_shift)

                    elif self._shift_type == "amplitude_change":
                        # Either we use the same shifts for all channels or we are sampling it
                        if self._same_shift:
                            channel_scalar = self._scalar_multi
                        else:
                            abs_shift = self.rng.uniform(self._scalar_multi * 0.75, self._scalar_multi)
                            channel_scalar = abs_shift * self.rng.choice([-1, 1])

                        new_arr = ch * channel_scalar
                    elif self._shift_type in ("theta_bandpass", "alpha_bandpass", "beta_bandpass", "delta_bandpass",
                                              "gamma_bandpass"):
                        new_arr = self._remove_frequency_band(data=ch)
                    elif self._shift_type == "baseline_drift":

                        if self._same_shift:
                            new_arr = ch + self._drift
                        else:
                            new_arr = self._add_baseline_drift(data=ch)

                    elif self._shift_type == "timewarp":
                        if self._same_shift:
                            new_arr = self._timewarp(data=ch, warp_signal=self._warp_signal)
                        else:
                            new_arr = self._timewarp(data=ch)
                    elif self._shift_type == "alpha_peak_shift":
                        new_arr = self._frequency_shift_channel(data=ch)
                    else:
                        raise ValueError("Unrecognized shift type")

                    # if self._plot_difference:
                    #     self.plot_shifted(original_data=ch, shifted_data=new_arr)

                else:
                    new_arr = ch

                altered_array[i][j] = new_arr

            # Plot psd

        # if self._shift_intensity == 1.0:
        #     self.plot_difference(org=data[2], new=altered_array[2], psd=True)
        #
        return altered_array

    def _channel_randomizing(self, data, channels_to_rotate):

        if self._shift_intensity == 0.1:
            channels_to_rotate = np.append(channels_to_rotate, [0])

        # Initialize an array to store the altered EEG data
        altered_array = np.zeros_like(data)

        # Compute a permutation of the channels to rotate.
        # This permutation will be applied identically for every subject.
        perm = self.rng.permutation(channels_to_rotate)

        all_not_rotated = True
        while all_not_rotated:
            # Check that no channel remains in its original position.
            if np.all(perm != channels_to_rotate):
                all_not_rotated = False
            else:
                perm = self.rng.permutation(channels_to_rotate)

        channels_not_rotated = np.setdiff1d(np.arange(19), channels_to_rotate)

        # For each subject, reorder the selected channels using the same permutation.
        n_subjects = data.shape[0]
        for i in range(n_subjects):
            # Replace the channels specified in channels_to_rotate with those in the permuted order.
            altered_array[i, channels_to_rotate, :] = data[i, perm, :]
            # Copy the remaining channels as they are.
            altered_array[i, channels_not_rotated, :] = data[i, channels_not_rotated, :]

        return altered_array

    def _frequency_shift_channel(self, data: np.ndarray) -> np.ndarray:
        """
        Shifts the spectral energy of the alpha band (8-12 Hz) in a single EEG channel,
        preserving the total energy. This simulates a shift in the dominant alpha peak.

        Parameters
        ----------
        data : np.ndarray
            A 1D array representing one EEG channel.

        Returns
        -------
        np.ndarray
            The EEG channel after applying the alpha frequency shift.
        """
        from scipy.signal import butter, filtfilt, hilbert

        # Get the sampling rate from self._eeg_info.
        sampling_rate = self._eeg_info['sfreq']
        n = len(data)

        # Define the alpha band explicitly.
        l_freq = 8.0  # lower bound of alpha
        h_freq = 12.0  # upper bound of alpha

        # Design a 4th-order Butterworth bandpass filter for the alpha band.
        nyq = 0.5 * sampling_rate
        low = l_freq / nyq
        high = h_freq / nyq
        order = 4
        b, a = butter(order, [low, high], btype='band')

        # Extract the alpha band component using zero-phase filtering.
        band_signal = filtfilt(b, a, data)

        # Compute the residual (signal outside the alpha band).
        residual = data - band_signal

        # Compute the analytic signal of the alpha component.
        analytic_band = hilbert(band_signal)

        # Create a time vector in seconds.
        t = np.arange(n) / sampling_rate

        # Choose a frequency shift for the alpha band:
        # Sample a random shift between 0 and max_shift_hz, and randomly assign a sign.

        if self._same_shift:
            shift_val = self._peak_shift
        else:
            shift_val = self.rng.uniform(self._peak_shift * 0.5, self._peak_shift * 1.25) * self.rng.choice([-1, 1])

        # Apply the frequency shift by modulating the analytic signal with a complex exponential.
        # This shifts every frequency component in the alpha band by shift_val.
        shifted_band = np.real(analytic_band * np.exp(1j * 2 * np.pi * shift_val * t))

        # Reconstruct the final signal by combining the shifted alpha component with the residual.
        shifted_data = residual + shifted_band

        return shifted_data

    def _calculate_warp_signal(self, n_times: int) -> np.ndarray:
        """
        Calculates a warp signal (new time indices) for a given number of time points.

        Parameters
        ----------
        n_times : int
            Number of time points.

        Returns
        -------
        np.ndarray
            A 1D array representing the new time indices.
        """
        # Scaling factor used for toning down the warp signal to avoid extreme warping
        scaling_factor = self.rng.uniform(0.5, 1.0)

        # Create an array representing the original time indices
        t_original = np.arange(n_times)

        # Randomly sample frequency and phase values
        freq = self.rng.uniform(0.01, 0.2)
        phase = self.rng.uniform(0, 2 * np.pi)

        max_warp_ratio = self.rng.uniform(self._max_warp_ratio * 0.75, self._max_warp_ratio)

        # Compute the warp offsets:
        # We compute a sine wave based on the random frequency and phase.
        # The sine wave oscillates between -1 and 1, so after multiplying by warp_amplitude,
        # the offsets vary between -max_warp_ratio and +max_warp_ratio.
        # Dividing t_original by n_times scales the sine function so that its period
        # is relative to the length of the data.
        warp_offsets = max_warp_ratio * np.sin(2 * np.pi * freq * t_original / n_times + phase)

        # Compute the new time indices (w_t) by adding the warp offsets (scaled by n_times) to the original time.
        # Multiplying by n_times scales the fractional offsets into actual index shifts.
        # w_t = t_original + warp_offsets * n_times
        w_t = t_original + warp_offsets * (n_times * scaling_factor)

        return w_t

    def _timewarp(self, data: np.ndarray, warp_signal: np.ndarray = None) -> np.ndarray:
        """
        Applies non-linear time warping to a single EEG channel.

        Parameters
        ----------
        data : np.ndarray
            A 1D array representing one EEG channel.
        warp_signal : np.ndarray, optional
            A precomputed warp signal. If provided, it is used for all channels.

        Returns
        -------
        np.ndarray
            The warped channel.
        """
        # Get the number of time points
        n_times = data.shape[0]

        # If warp_signal is not provided, calculate it
        if warp_signal is None:
            warp_signal = self._calculate_warp_signal(n_times)

        # Create an interpolation function based on the original data.
        # The interpolation is linear, and it will use the first and last values
        # as fill values if the new time indices are outside the original range.
        t_original = np.arange(n_times)
        f = interp1d(t_original, data, kind='linear', bounds_error=False, fill_value=(data[0], data[-1]))

        # Use the warp signal (the new time indices) to interpolate the data.
        # This effectively "remaps" the original data according to the warp signal.
        warped_channel = f(warp_signal)
        return warped_channel

    def _add_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """
        Adds a combination of low-frequency sinusoids with random frequencies, phases,
        and weights to simulate realistic baseline drifts. A fixed random seed ensures
        reproducible results.

        Parameters
        ----------
        data : np.ndarray
            The EEG data for a single channel (1D array of shape (n_times,)).

        Returns
        -------
        np.ndarray
            The augmented EEG data with baseline drift added (same shape as input).
        """
        data_drifted = data.copy()

        # Get the number of time points (ensure it's an integer)
        n_times = data_drifted.shape[0]

        # Create a time axis over which to define our sinusoidal waves
        time_axis = np.linspace(0, 2 * np.pi, n_times)

        # Overall amplitude of the drift
        drift_amplitude = self._max_drift_amplitude

        # Generate multiple sinusoids with random frequencies and phases
        sinusoids = []
        for _ in range(self._num_sinusoids):
            freq = self.rng.uniform(0.01, 0.2)  # random low frequency
            phase = self.rng.uniform(0, 2 * np.pi)  # random starting phase
            wave = np.sin(freq * time_axis + phase)
            sinusoids.append(wave)

        # Combine the sinusoids with random weights
        combined = np.zeros_like(time_axis)
        for wave in sinusoids:
            weight = self.rng.uniform(0.5, 1.5)
            combined += weight * wave

        # Add the baseline drift to the original signal
        data_drifted += drift_amplitude * combined

        return data_drifted

    def _calculate_drift_signal(self, n_times: int) -> np.ndarray:
        """
        Computes a baseline drift signal as a combination of low-frequency sinusoids.

        Parameters
        ----------
        n_times : int
            Number of time points in the signal.

        Returns
        -------
        drift_signal : np.ndarray
            A 1D array of shape (n_times,) representing the computed drift.
        """
        # Create time axis over which to define sinusoids
        time_axis = np.linspace(0, 2 * np.pi, n_times)
        # Overall amplitude of the drift
        drift_amplitude = self._max_drift_amplitude

        # Generate and combine multiple sinusoids
        combined = np.zeros_like(time_axis)
        for _ in range(self._num_sinusoids):
            freq = self.rng.uniform(0.1, 0.75)  # Random low frequency
            phase = self.rng.uniform(0, 2 * np.pi)  # Random phase
            wave = np.sin(freq * time_axis + phase)
            # Weight each wave with a random factor
            weight = self.rng.uniform(0.5, 1.0)
            combined += weight * wave

        # Scale the combined waveform by the drift amplitude
        drift_signal = drift_amplitude * combined
        return drift_signal

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
        if isinstance(shift, float):
            shift = int(shift)

        if data.ndim == 1:
            return np.roll(data, shift)
        elif data.ndim == 2:
            # Assuming shape is (n_channels, n_times)
            return np.roll(data, shift, axis=1)
        elif data.ndim == 3:
            # Assuming shape is (n_samples, n_channels, n_times)
            return np.roll(data, shift, axis=2)

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

    def _remove_frequency_band(self, data):
        """

        Parameters
        ----------
        data: single EEG channel

        Returns
        -------

        """
        h_freq, l_freq = self._get_freq()
        # Ensure data is in 2D format
        eeg_data_2d = data[np.newaxis, :]  # Shape becomes (1, 2000)
        # Create MNE info structure
        info = mne.create_info(ch_names=['EEG 001'], sfreq=self._eeg_info['sfreq'],
                               ch_types=['eeg'])
        # Create Raw object
        raw = mne.io.RawArray(eeg_data_2d, info, verbose=False)
        # High-pass filter to remove frequencies below 12 Hz
        # raw_high_pass = raw.copy().filter(l_freq=l_freq, h_freq=None, verbose=False)
        #
        # # Low-pass filter to remove frequencies above 8 Hz
        # raw_low_pass = raw.copy().filter(l_freq=None, h_freq=h_freq, verbose=False)
        #
        # # Combine the filtered signals by adding the data
        # combined_data = raw_high_pass.get_data() + raw_low_pass.get_data()
        if "alpha" in self._shift_type:
            combined_data = raw.copy().filter(l_freq=l_freq,
                                              h_freq=h_freq,
                                              fir_design='firwin',
                                              filter_length=353,
                                              verbose=False).get_data()
        else:
            combined_data = raw.copy().filter(l_freq=l_freq,
                                              h_freq=h_freq,
                                              fir_design='firwin',
                                              filter_length='auto',
                                              verbose=False).get_data()

        return combined_data

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
        augment_order = np.arange(19)

        # Remove Cz and Pz from the list of channels to augment
        augment_order = np.delete(augment_order, [17, 18])

        # Shuffle the order of the channels, so that the channels to augment are random, but reproducible
        self.rng.shuffle(augment_order)

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
            return 7.5, 12.5
        elif freq == "theta":
            return 3.5, 8.5
        elif freq == "delta":
            return 0.5, 4.5
        elif freq == "beta":
            return 11.5, 30.5
        elif freq == "gamma":
            return 29.5, 80.5
        else:
            raise ValueError(f"Frequency band: {freq} is not recognized")

    def plot_difference(self, org, new, psd=False):
        info_object = self._eeg_info
        montage = mne.channels.make_standard_montage('standard_1020')
        info_object.set_montage(montage)
        raw = mne.io.RawArray(org, info_object, verbose=False)
        raw_new = mne.io.RawArray(new, info_object, verbose=False)

        if psd:
            # Plot the power spectral density of the EEG data
            raw.compute_psd().plot(average=True)
            raw.plot(block=True)

            raw_new.compute_psd().plot()
            raw_new.plot(block=True)
            raw_new.compute_psd().plot(average=True)
            raw_new.plot(block=True)

        else:
            raw.plot(block=True, scalings='auto')
            raw_new.plot(block=True, scalings='auto')

        plt.close()

    @staticmethod
    def plot_shifted(original_data, shifted_data):
        # 3) Plot original vs drifted for each channel
        time_axis = np.arange(6000)

        offset = 3

        # Create a figure, with two channels in the same plot
        # No need for subplots, since we only have two channels
        plt.plot(time_axis, original_data + offset, label='Original', alpha=0.7)
        plt.plot(time_axis, shifted_data, label='Drifted', alpha=0.7)
        plt.ylabel("Ch 0")
        plt.ylabel("Time")
        plt.legend(loc='upper right')

        plt.suptitle("Dataset shift Example on 1 Channels")
        plt.tight_layout()
        plt.close()

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    def __len__(self) -> int:  # type: ignore[no-any-return]
        return self._x.size()[0]

    def __getitem__(self, index):
        age_tensor = self.ages[index].clone().detach().view(1, -1)
        age_tensor = age_tensor.expand(1, self._x[index].shape[1])
        combined_data = torch.cat((self._x[index], age_tensor), dim=0)

        return combined_data, self._y[index], index

    def get_subject_keys_from_indices(self, indices):
        return [self._subject_keys[i] for i in indices]
