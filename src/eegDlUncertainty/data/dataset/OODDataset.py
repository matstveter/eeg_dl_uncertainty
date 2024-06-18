import abc
import json
import os
from abc import abstractmethod

import matplotlib
import mne
import matplotlib.pyplot as plt

import numpy
import numpy as np
import pandas as pd

from eegDlUncertainty.data.dataset.misc_classes import AgeScaler


class BaseDataset(abc.ABC):
    def __init__(self, *, dataset_version, num_seconds_eeg, age_scaling):
        self._dataset_version = dataset_version
        self._num_seconds_eeg = num_seconds_eeg
        self._age_scaling = age_scaling
        self._num_epochs = 1

        try:
            config = self._read_config(json_path=os.path.join(os.path.dirname(__file__),
                                                              "ood_dataset_config.json"))[self.__class__.__name__]
        except KeyError:
            raise KeyError(f"Key :{self.__class__.__name__} does not exist in the config!")

        self._dataset_path = os.path.join(config['base_dataset_path'], f"numpy_v{dataset_version}")

        # Read the files of the subjects that are as numpy files
        self._subjects = self._get_potential_subjects()

        # Read the label file
        if self.__class__.__name__ == "MPILemonDataset":
            self._labels = self._read_label_file(path=os.path.join(config['label_dir'], "participants.csv"))
            prep_path = os.path.join(self._dataset_path, "data_processing_mpi_lemon.json")
        else:
            self._labels = self._read_label_file(path=os.path.join(config['label_dir'], "participants.tsv"))

            if self.__class__.__name__ == "TDBrainDataset":
                prep_path = os.path.join(self._dataset_path, f"data_processing_tdbrain.json")
            else:
                prep_path = os.path.join(self._dataset_path, f"data_processing_greek_eeg.json")

        self._preprocessing = self._read_config(prep_path)['preprocessing']
        self._ageScaler = None

        self._num_channels = len(self.get_eeg_info()['ch_names'])
        self._eeg_time_points = self._preprocessing['sfreq'] * self._num_seconds_eeg

        # This function checks the labels, checks subjects and transforms the self._labels to a dict with only needed
        # info
        self.prepare_dataset()

    @property
    def subjects(self):
        return self._subjects

    def load_eeg_data(self, plot=False):

        data = numpy.zeros(shape=(len(self.subjects) * self._num_epochs, self._num_channels, self._eeg_time_points))
        for i, sub in enumerate(self._subjects):
            sub_path = f"{self._dataset_path}/{sub}.npy"
            npy_data = numpy.load(sub_path)

            # Normalize the eeg signal
            npy_data = self.__normalize_data(x=npy_data)
            raw = mne.io.RawArray(data=npy_data, info=self.get_eeg_info(), verbose=False)

            if plot:
                matplotlib.use("TKAgg")
                raw.plot(block=True)
                plt.close()

            epochs = mne.make_fixed_length_epochs(raw=raw, duration=self._num_seconds_eeg,
                                                  preload=True, verbose=False)
            epoch_numpy_data = epochs.get_data(copy=False)
            npy_data = epoch_numpy_data[0:self._num_epochs, :, :]

            cur_index = 0
            for j in range((i * self._num_epochs), (i * self._num_epochs) + self._num_epochs):
                data[j] = npy_data[cur_index]
                cur_index += 1

        return data

    def load_targets(self):
        """ This function loads the target class from the label dictionary, and repaeats the calss labels num_epochs
        times. It will be the same if num_epochs is 1.

        Returns
        -------
        class_labels: np.ndarray
            labels for the subjects
        """

        class_labels = np.array([self._labels[sub]['class_label'] for sub in self._subjects])
        class_labels = np.repeat(class_labels, self._num_epochs)
        return class_labels

    def load_ages(self, add_noise=False):
        """ Load age of the subjects.

        This function receives a tuple of subject IDs, it loops through these subjects and extracts the age from
        the loaded dictionary.

        Parameters
        ----------
        add_noise

        Returns
        -------
        data: np.ndarray
            structure = [60, 65, 70, ..., n_subjects], shape=(n_subjects, 1)
        """
        transformed_ages = self._ageScaler.transform(sub_ids=self.subjects, add_noise=add_noise)

        if self._num_epochs == 1:
            return transformed_ages
        else:
            return np.repeat(transformed_ages, self._num_epochs)

    @staticmethod
    def __normalize_data(x):
        """
        Normalize the data to the range [-1, 1].
        If method is 'channel', normalization is done channel-wise.
        If method is 'subject', normalization is done across all channels for the subject.
        """
        x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-8)
        return x

    @staticmethod
    def _read_config(json_path: str):
        with open(json_path) as json_reader:
            return json.load(json_reader)

    @staticmethod
    def _read_label_file(path):
        if path[-3:] == "csv":
            sep = ","
        else:
            sep = "\t"
        return pd.read_csv(path, sep=sep)

    def _get_potential_subjects(self):
        files = sorted(os.listdir(path=self._dataset_path))
        # Get only numpy files, and remove the numpy ending
        return [sub.split(sep=".")[0] for sub in files if ".npy" in sub]

    def _set_age_scaler(self):
        self._ageScaler = AgeScaler(dataset_dict=self._labels, scaling_type=self._age_scaling)

    @abstractmethod
    def prepare_dataset(self):
        pass

    def get_eeg_info(self):
        """
        Generate EEG info object with predefined channel names and sampling frequency.

        This private method creates an MNE Info object that contains information about the EEG setup,
        including channel names and the sampling frequency. The channel names are hard-coded for a 19-channel
        EEG system using a specific naming convention. The sampling frequency is calculated based on the
        'high_freq' and 'nyquist' values specified in the `_preprocessing_steps` attribute of the instance.

        Note that this method is specifically designed for a 19-channel EEG system and may not be suitable for
        other configurations without modification.

        Returns
        -------
        mne.Info
            An MNE Info object containing the EEG setup information, including channel names and sampling frequency.

        Notes
        -----
        - The channel names are hard-coded for a specific 19-channel EEG system setup and include the following:
          ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'O1-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG',
           'P4-AVG', 'O2-AVG', 'F7-AVG', 'T3-AVG', 'T5-AVG', 'F8-AVG', 'T4-AVG', 'T6-AVG',
           'FZ-AVG', 'CZ-AVG', 'PZ-AVG'].
        - The sampling frequency (`sfreq`) is calculated from the preprocessing steps, specifically from
          the 'high_freq' and 'nyquist' parameters. Ensure these are correctly set in `_preprocessing_steps`.
        """
        ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4',
                    'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
                    'Fz', 'Cz', 'Pz']
        sfreq = self._preprocessing['sfreq']  # From the paper
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")


class TDBrainDataset(BaseDataset):

    def prepare_dataset(self) -> None:
        """
        Prepare the dataset by filtering and processing participant labels and ages.

        This method performs the following steps:
        1. Filters the `_labels` DataFrame to include only the participants listed in `_subjects`.
        2. Selects the 'participants_ID' and 'age' columns.
        3. Drops rows where 'age' is missing.
        4. Converts 'age' values to integers.
        5. Creates a dictionary with 'participants_ID' as keys and another dictionary containing 'age' and a
         'class_label' of 0 as values.
        6. Updates the `_labels` attribute with this dictionary.
        7. Updates the `_subjects` attribute with the filtered list of 'participants_ID'.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Filter the labels to include only the subjects of interest
        label = self._labels[self._labels['participants_ID'].isin(self._subjects)].copy()
        # Select only 'participants_ID' and 'age' columns
        filtered_labels = label[['participants_ID', 'age']]
        # Drop rows where 'age' is missing
        filtered = filtered_labels.dropna(subset=['age'])
        # Convert 'age' to integers
        filtered.loc[:, 'age'] = filtered['age'].apply(lambda x: int(str(x).split(",")[0]))

        # Create a dictionary with 'participants_ID' as keys and 'age' as values
        data_dict = filtered.set_index('participants_ID')['age'].to_dict()

        # Create a final dictionary with 'participants_ID' as keys and a dictionary with 'age' and 'class_label' as
        # values
        final_dict = {}
        for k, v in data_dict.items():
            final_dict[k] = {'age': v,
                             'class_label': 0}

        # Update the object's _labels attribute with the final dictionary
        self._labels = final_dict
        # Update the object's _subjects attribute with the list of filtered 'participants_ID's
        self._subjects = filtered['participants_ID'].to_list()


class GreekEEGDataset(BaseDataset):

    def prepare_dataset(self) -> None:
        """
        Prepares and filters the dataset for subjects of interest.

        This method filters the labels to include only the subjects of interest, selects relevant columns,
        and creates a dictionary mapping each participant ID to their age and class label. It updates the
        object's `_labels` attribute with this dictionary and the `_subjects` attribute with the list of
        filtered participant IDs.

        Notes
        -----
        The class labels are assigned as follows:
        - 'C' (Control) is labeled as 0 (healthy).
        - 'A' (Alzheimer) and 'F' (Frontotemporal dementia) are labeled as 2 (dementia).

        Raises
        ------
        KeyError
            If required columns ('participant_id', 'Age', 'Group') are not found in the labels DataFrame.

        Examples
        --------
        >>> obj = MyClass()
        >>> obj.prepare_dataset()
        """
        # Filter the labels to include only the subjects of interest
        label = self._labels[self._labels['participant_id'].isin(self._subjects)].copy()

        # Select only 'participant_id', 'Age', and 'Group' columns
        filtered_labels = label[['participant_id', 'Age', 'Group']]

        # Loop through the DataFrame and create a dictionary
        final_dict = {}
        for _, row in filtered_labels.iterrows():
            # If Group is 'C' (Control), set the class to healthy (0)
            if row['Group'] == "C":
                class_label = 0
            else:
                # Else, the class is 'A' (Alzheimer) or 'F' (Frontotemporal dementia), so set class to dementia (2)
                class_label = 2
            final_dict[row['participant_id']] = {'age': row['Age'], 'class_label': class_label}

        # Update the object's _labels attribute with the new dictionary
        self._labels = final_dict

        # Update the object's _subjects attribute with the list of filtered 'participant_id's
        self._subjects = filtered_labels['participant_id'].to_list()


class MPILemonDataset(BaseDataset):

    def prepare_dataset(self):
        """
        Prepares the dataset by filtering and processing label information.

        This method filters the `_labels` DataFrame to include only the subjects
        specified in `_subjects`. It then processes the 'Age' column to calculate
        the mean age for age ranges, and assigns a default class label. The resulting
        information is stored in a dictionary format within the `_labels` attribute,
        and the `_subjects` attribute is updated with the filtered IDs.

        Returns
        -------
        None
            This method updates the `_labels` and `_subjects` attributes in place.

        Notes
        -----
        The 'Age' column in the `_labels` DataFrame is expected to be a string
        representing a range (e.g., '35-40'). The method calculates the mean age
        for these ranges.
        """
        label = self._labels[self._labels['ID'].isin(self._subjects)].copy()
        # Select only 'participant_id', 'Age', and 'Group' columns
        filtered_labels = label[['ID', 'Age']]

        # Loop through the DataFrame and create a dictionary
        final_dict = {}
        for _, row in filtered_labels.iterrows():
            # Split the ages as this is a str on the format 35-40 or 65-70 or similar
            ages = row['Age'].split('-')
            # This is typically 65-70, so this will return 67
            age = int((int(ages[0]) + int(ages[1]))/2)
            # Healthy
            class_label = 0
            final_dict[row['ID']] = {'age': age, 'class_label': class_label}

        # Update the object's _labels attribute with the new dictionary
        self._labels = final_dict

        # Update the object's _subjects attribute with the list of filtered 'participant_id's
        self._subjects = filtered_labels['ID'].to_list()
