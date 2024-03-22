import json
import os
from typing import Any, Dict, List, Tuple
import numpy
import mne
import numpy as np
import torch.nn.functional
from tqdm import tqdm


class CauEEGDataset:

    def __init__(self, dataset_version, targets: str, eeg_len_seconds: int, prediction_type: str, which_one_vs_all: str,
                 pairwise: str, use_predefined_split: bool = True, num_channels: int = 19, epochs: int = 1):
        # Read in the dataset config
        config = self._read_config(json_path=os.path.join(os.path.dirname(__file__), "dataset_config.json"))

        if targets in ("dementia", "dementia-no-overlap", "abnormal", "abnormal-no-overlap"):
            loaded_dict = self._read_config(os.path.join(config.get('label_dir'), f"{targets}.json"))
        else:
            raise KeyError("Unrecognized label type: [dementia, dementia-no-overlap, abnormal, abnormal-no-overlap)")

        self._task_name = loaded_dict['task_name']
        self._class_name_to_label = loaded_dict['class_name_to_label']

        # Prediction related
        self._prediction_type = prediction_type
        self._which_one_vs_all = which_one_vs_all
        self._pairwise = pairwise

        if self._prediction_type == "normal" and len(self._class_name_to_label) == 3:
            self._num_classes = 3
        else:
            self._num_classes = 1

        if use_predefined_split:
            self._train_split = self._get_participant_info(dataset_split=loaded_dict['train_split'])
            self._val_split = self._get_participant_info(dataset_split=loaded_dict['validation_split'])
            self._test_split = self._get_participant_info(dataset_split=loaded_dict['test_split'])
        else:
            raise NotImplementedError("Missing ")

        self._merged_splits = self._merge_splits(train_split=self._train_split,
                                                 val_split=self._val_split,
                                                 test_split=self._test_split)

        # Get the suggested dataset path from the config file
        suggested_dataset = os.path.join(config.get('base_dataset_path'), f"caug_numpy_v{dataset_version}")

        # Check if the path or file exists
        if os.path.exists(suggested_dataset):
            self._dataset_path = suggested_dataset
            # Read the config and only extract the preprocessing steps from the config file
            self._preprocessing_steps = self._read_config(os.path.join(
                suggested_dataset, "data_processing.json"))['preprocessing']
        else:
            raise KeyError(f"Specified dataset version: {dataset_version} does not exist, "
                           f"potential datasets are: {os.listdir(config.get('base_dataset_path'))}")

        # Calculate the number of actual time steps based on seconds, and then the current sampling frequency
        self._eeg_len = (eeg_len_seconds *
                         (self._preprocessing_steps['high_freq'] * self._preprocessing_steps['nyquist']))
        self._epochs = epochs

        # Set default to 19 channels
        self._num_channels = num_channels

    @property
    def eeg_len(self):
        return self._eeg_len

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def _merge_splits(train_split: Dict[str, Dict[str, Any]], val_split: Dict[str, Dict[str, Any]],
                      test_split: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Merges the train, validation, and test splits into a single dictionary after
        checking for overlaps to avoid data leakage.

        Parameters:
        - train_split: dict, the training split with subjects or data points.
        - val_split: dict, the validation split with subjects or data points.
        - test_split: dict, the test split with subjects or data points.

        Returns:
        - merged_splits: dict, a dictionary containing all the non-overlapping entries
          from the train, validation, and test splits.

        Raises:
        - AssertionError: if there is an overlap between any of the splits, indicating
          potential data leakage.
        """
        # Checks for overlapping subjects in the various sets
        assert not set(tuple(train_split)) & set(val_split), \
            "Data Leakage: Found subjects in both train and val"
        assert not set(tuple(train_split)) & set(test_split), \
            "Data Leakage: Found subjects in both train and test"
        assert not set(tuple(val_split)) & set(test_split), \
            "Data Leakage: Found subjects in both test and val"

        # Make a combined dictionary
        return {**train_split, **val_split, **test_split}

    @staticmethod
    def _get_participant_info(dataset_split: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
            Extracts and organizes participant information from a given dataset split.

            Iterates through each entry in the dataset split, which is expected to be a list of dictionaries,
            each representing a participant's information. It organizes this information into a new dictionary
            where each key is the participant's serial number, and the value is the participant's
            information as a dictionary.

            Parameters
            ----------
            dataset_split : List[Dict[str, Any]]
                A list of dictionaries, where each dictionary contains information about a participant in the dataset.
                Each dictionary is expected to have a key named 'serial' which is used to identify the participant.

            Returns
            -------
            Dict[str, Dict[str, Any]]
                A dictionary where the keys are the serial numbers of the participants, and the values are the
                dictionaries containing the participants' information from the input list.
        """
        subject_dict = {}
        for sub in dataset_split:
            subject_dict[sub['serial']] = sub
        return subject_dict

    def get_split_statistics(self, split: str):
        """
        Retrieves and prints statistical summaries for a specified data split.

        This function selects one of the predefined data splits ('train', 'val', 'test') and computes
        statistical summaries for that split using the nested `summarize` function. The summaries include
        the total number of subjects, average age, class distribution, and class percentages. It then prints
        these statistics in a readable format. The `summarize` function, defined within this method, is responsible
        for calculating these statistics.

        Parameters
        ----------
        split : str
            The name of the data split to compute statistics for. Expected values are 'train', 'val', or 'test'.

        Raises
        ------
        KeyError
            If the `split` parameter does not match one of the accepted keys ['train', 'val', 'test'].

        See Also
        --------
        summarize : The nested function that performs the actual computation of statistics.

        Notes
        -----
        - The function assumes that the data splits ('_train_split', '_val_split', '_test_split') have been
          initialized elsewhere within the class and contain lists of dictionaries with subject information.
        - The statistical summaries are printed directly by this function and are not returned.

        Examples
        --------
        Assuming an instance `instance` of the class has been created and initialized:

        >>> self.get_split_statistics('train')
        Statistic for train split
        Total Subjects: 150
        Avg. Age: 27.5
        Class Name: {0: 'Class 0', 1: 'Class 1', ...}
        Class Counts: {0: 75, 1: 75}
        Class %: {0: 50.0, 1: 50.0}
        """

        def summarize(subject_split: Dict[str, Any]) -> Dict[str, Any]:
            """
             Generates a summary of subjects based on their split information.

             This function takes a dictionary where each key is a subject identifier and its value is another
             dictionary containing the subject's information, including their age and class label. It produces a
             summary that includes the total number of subjects, the average age of subjects, counts and percentages
             of subjects in each class, and maps class names to class labels.

             Parameters
             ----------
             subject_split : Dict[str, Any]
                 A dictionary with subject identifiers as keys and dictionaries containing subject information
                 (e.g., age, class_label) as values.

             Returns
             -------
             Dict[str, Any]
                 A dictionary containing the following keys:
                 - 'total_subjects': The total number of subjects in the split.
                 - 'average_age': The average age of subjects in the split.
                 - 'class_counts': A dictionary with class labels as keys and counts of subjects in each class
                 as values.
                 - 'class_percentage': A dictionary with class labels as keys and the percentage of subjects in each
                   class relative to the total number of subjects as values.
                 - 'class_name': A mapping of class names to class labels (assumed to be provided by an instance
                 variable).
             """
            summary = {'total_subjects': len(subject_split),
                       'total_age': 0,
                       'class_counts': {},
                       'class_percentage': {},
                       'class_name': self._class_name_to_label}

            # Loop through the subjects in the dictionary and extract class label
            for sub_id, sub_info in subject_split.items():
                class_label = sub_info['class_label']
                if class_label in summary['class_counts']:
                    summary['class_counts'][class_label] += 1
                else:
                    summary['class_counts'][class_label] = 1

                # Accumulate age for average calculation
                summary['total_age'] += sub_info['age']

            # Calculate average age
            if summary['total_subjects'] > 0:
                summary['average_age'] = summary['total_age'] / summary['total_subjects']
            else:
                summary['average_age'] = 0

            # Remove total_age from summary as it's no longer needed
            del summary['total_age']

            # Calculate the percentage of each class
            for k, v in summary['class_counts'].items():
                summary['class_percentage'][str(k)] = (v / summary['total_subjects']) * 100

            return summary

        if split == "train":
            data_summary = summarize(subject_split=self._train_split)
        elif split == "val":
            data_summary = summarize(subject_split=self._val_split)
        elif split == "test":
            data_summary = summarize(subject_split=self._test_split)
        else:
            raise KeyError("Split is not one of the accepted keys ['train', 'val', 'test']")

        print(f"Statistic for {split} split")
        print(f"Total Subjects: {data_summary['total_subjects']}")
        print(f"Avg. Age: {data_summary['average_age']}")
        print(f"Class Name: {data_summary['class_name']}")
        print(f"Class Counts: {data_summary['class_counts']}")
        print(f"Class %: {data_summary['class_percentage']}")

    def get_splits(self) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        """
            Retrieves the tuples of subject identifiers for the training, validation, and test splits.

            This method ensures that there is no overlap between the subjects in the training, validation,
            and test sets to prevent data leakage. It performs checks to assert that the subjects are
            exclusive to each split and raises an AssertionError with a descriptive message if any overlap is found.

            Returns
            -------
            Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]
                A tuple containing three tuples: the first for training subjects, the second for validation subjects,
                and the third for test subjects. Each inner tuple contains subject identifiers as strings.

            Raises
            ------
            AssertionError
                If any subject appears in more than one of the training, validation, or test splits,
                indicating potential data leakage.
        """
        train_subjects = tuple(self._train_split)
        val_subjects = tuple(self._val_split)
        test_subjects = tuple(self._test_split)

        if self._prediction_type == "pairwise":
            train_subjects = self._get_pairwise_dataset(data_set=train_subjects)
            val_subjects = self._get_pairwise_dataset(data_set=val_subjects)
            test_subjects = self._get_pairwise_dataset(data_set=test_subjects)

        return train_subjects, val_subjects, test_subjects

    def load_targets(self, subjects: Tuple[str, ...]) -> numpy.ndarray:  # type: ignore[type-arg, return]

        class_labels = np.array([self._merged_splits[sub]['class_label'] for sub in subjects])

        if self._task_name == 'CAUEEG-Abnormal benchmark':
            return class_labels
        elif self._task_name == 'CAUEEG-Dementia benchmark':
            if self._prediction_type == "normal":
                one_hot_class_labels: numpy.ndarray = torch.nn.functional.one_hot(  # type: ignore[type-arg]
                    torch.from_numpy(class_labels), num_classes=len(self._class_name_to_label)).numpy()
                return one_hot_class_labels
            elif self._prediction_type == "one_vs_all":
                if self._which_one_vs_all == "normal":
                    class_lab = 0
                elif self._which_one_vs_all == "mci":
                    class_lab = 1
                elif self._which_one_vs_all == "dementia":
                    class_lab = 2
                else:
                    raise KeyError("Unrecognized class name for one-vs-all predictions")
                class_labels = np.array([1 if c_lab == class_lab else 0 for c_lab in class_labels])

                return class_labels
            elif self._prediction_type == "pairwise":
                class_labels = np.array([0 if self._pairwise[0] == self._merged_splits[sub]['class_name'].lower() else 1
                                         for sub in subjects])
                return class_labels
        else:
            raise KeyError(f"Unrecognized task name: {self._task_name}")

    def load_eeg_data(self, subjects: Tuple[str, ...], plot: bool = False) -> numpy.ndarray:  # type: ignore[type-arg]
        """
        Load EEG data for a given list of subjects and optionally plot the raw EEG data.

        This method initializes an empty data array based on the number of epochs set for the instance.
        If the number of epochs is set to 1 or 0, it creates a 3D array with dimensions corresponding to
        the number of subjects, the number of EEG channels, and the length of the EEG data. For other
        values of epochs, it raises a NotImplementedError.

        As the data for each subject is loaded, a progress bar updates to reflect the process. If the
        'plot' parameter is set to True, the method plots the raw EEG data for each subject using MNE's
        plotting capabilities.

        Parameters
        ----------
        subjects : list
            A list of subject identifiers whose EEG data is to be loaded.
        plot : bool, optional
            A flag that, if set to True, plots the raw EEG data for each subject as it is loaded. The default is False.

        Returns
        -------
        numpy.ndarray
            A numpy array containing the loaded EEG data. The array will have a shape of
            (len(subjects), num_channels, eeg_len) if epochs are set to 1 or 0, or a shape of
            (len(subjects), epochs, num_channels, eeg_len) for other epoch values, but the latter
            case is not yet implemented.

        Raises
        ------
        NotImplementedError
            If the number of epochs is set to a value other than 1 or 0, indicating that the
            method for handling multiple epochs is not yet implemented.

        Notes
        -----
        - The EEG data is expected to be stored in a numpy file (.npy) with a name matching the subject identifier
          in a predefined dataset path.
        - The method uses a tqdm progress bar to visually indicate the progress of loading data.

        Examples
        --------
        >>> self.load_eeg_data(['subject1', 'subject2'], plot=True)
        # This will load the EEG data for 'subject1' and 'subject2', plot the raw data, and return the data array.
        """
        if self._epochs == 1 or self._epochs == 0:
            data = numpy.zeros(shape=(len(subjects), self._num_channels, self._eeg_len))
        else:
            # data = numpy.zeros(shape=(len(subjects), self._epochs, self._num_channels, self._eeg_len))
            # transform to mne raw then convert using the functionality from mne?
            raise NotImplementedError("EEG epochs set to other than 1, missing implementation!")

        # Progress bar
        pbar = tqdm(total=len(subjects), desc="Loading", unit="subjects", colour='green')
        # Loop through subject list
        for i, sub in enumerate(subjects):
            # Create subject numpy path
            subject_eeg_data_path = f"{self._dataset_path}/{sub}.npy"
            # Load the subject numpy file
            npy_data = numpy.load(subject_eeg_data_path)

            if self._eeg_len > npy_data.shape[1]:
                raise ValueError("Specified EEG length is longer than the actual recording.")
            else:
                npy_data = npy_data[:, 0: self._eeg_len]

            # Plotting function
            if plot:
                raw = mne.io.RawArray(data=npy_data, info=self.__get_eeg_info(), verbose=False)
                raw.plot(block=True)

            # npy_data = self.__normalize_data(data=npy_data, method='subject')

            # Add the data to the empty numpy array
            data[i] = npy_data
            pbar.update(1)

        # Close the progress bare, this could probably be avoided with a with statement instead, but...
        pbar.close()
        return data

    def __normalize_data(self, data, method='channel'):
        """
        Normalize the data to the range [-1, 1].
        If method is 'channel', normalization is done channel-wise.
        If method is 'subject', normalization is done across all channels for the subject.
        """
        if method == 'channel':
            # Channel-wise normalization
            min_val = np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
        elif method == 'subject':
            # Subject-wise normalization
            min_val = np.min(data)
            max_val = np.max(data)
        else:
            raise ValueError("Normalization method must be 'channel' or 'subject'.")

        # Normalize to [-1, 1]
        return 2 * (data - min_val) / (max_val - min_val) - 1


    @staticmethod
    def _read_config(json_path: str):
        with open(json_path) as json_reader:
            return json.load(json_reader)

    def __get_eeg_info(self):
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
        ch_names = ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'O1-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG',
                    'P4-AVG', 'O2-AVG', 'F7-AVG', 'T3-AVG', 'T5-AVG', 'F8-AVG', 'T4-AVG', 'T6-AVG',
                    'FZ-AVG', 'CZ-AVG', 'PZ-AVG']
        sfreq = self._preprocessing_steps['high_freq'] * self._preprocessing_steps['nyquist']
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    def _get_pairwise_dataset(self, data_set: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Filters a dataset tuple to include only the items that belong to a specified pair of classes.

        This method checks each item in the given dataset to determine if it belongs to one of two predefined
        classes specified in the object's `_pairwise` attribute. Only items that match these classes are included
        in the new dataset tuple that is returned.

        Parameters
        ----------
        data_set : Tuple[str, ...]
            A tuple containing the dataset items to be filtered. Each item is expected to be a string identifier
            that can be used to look up class information in the object's `_merged_splits` attribute.

        Returns
        -------
        Tuple[str, ...]
            A tuple containing the filtered set of dataset items. Each item in the tuple is a string identifier
            for items that belong to the specified pair of classes.

        Raises
        ------
        ValueError
            If the `_pairwise` attribute does not contain exactly two classes, indicating that the method cannot
            proceed with the filtering as it requires exactly two classes to compare against.

        Notes
        -----
        The method assumes that `self._pairwise` is a collection (e.g., list or set) containing the names of
        two classes to filter the dataset by. It also assumes that `self._merged_splits` is a dictionary with
        dataset item identifiers as keys and dictionaries as values, where each dictionary contains a 'class_name'
        key with a string value representing the item's class name.
        """
        new_set = []

        if len(self._pairwise) != 2:
            raise ValueError("The specified pair in the config file does not contain two classes!")

        for sub in data_set:
            if self._merged_splits[sub]['class_name'].lower() in self._pairwise:
                new_set.append(sub)

        return tuple(new_set)
