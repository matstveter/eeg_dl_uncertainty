import json
import os
from typing import Any, Dict, List, Tuple

import numpy
import mne
import numpy as np
import torch.nn.functional
from tqdm import tqdm

from eegDlUncertainty.data.dataset.misc_classes import AgeScaler, verify_split_subjects


class CauEEGDataset:

    def __init__(self, *, dataset_version, targets: str, eeg_len_seconds: int,
                 epochs: str, overlapping_epochs: bool, save_dir: str, use_predefined_split: bool = True,
                 num_channels: int = 19, age_scaling="Standard"):
        # Read in the dataset config
        config = self._read_config(json_path=os.path.join(os.path.dirname(__file__), "dataset_config.json"))

        if targets in ("dementia", "dementia-no-overlap", "abnormal", "abnormal-no-overlap"):
            loaded_dict = self._read_config(os.path.join(config.get('label_dir'), f"{targets}.json"))
        else:
            raise KeyError("Unrecognized label type: [dementia, dementia-no-overlap, abnormal, abnormal-no-overlap)")

        self._task_name = loaded_dict['task_name']
        self._class_name_to_label = loaded_dict['class_name_to_label']
        self._save_dir = save_dir

        # Prediction related
        self._name = "CAUEEG"

        self._num_classes = len(self._class_name_to_label)

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

        self._ageScaler = AgeScaler(dataset_dict=self._merged_splits, scaling_type=age_scaling)
        self._overlapping_epochs = overlapping_epochs
        self._eeg_len = int(eeg_len_seconds * self.get_eeg_info()['sfreq'])
        self._num_seconds = eeg_len_seconds

        maximum_epochs = self._preprocessing_steps['num_seconds_per_subject'] / self._num_seconds

        if epochs == "all":
            self._epochs = int(maximum_epochs)
            print(self._epochs)
        else:
            # In the case of spread or random the number of epochs is set to half of the maximum
            self._epochs = int(maximum_epochs / 2)

        if self._epochs > 1:
            self._num_val_epochs = 2
        else:
            self._num_val_epochs = 1

        self._epoch_structure = epochs
        self._eeg_info = self.get_eeg_info()

        if self._overlapping_epochs:
            duration = (self._epochs - 1) * (eeg_len_seconds / 2)
        else:
            duration = self._epochs * eeg_len_seconds

        assert duration <= self._preprocessing_steps['num_seconds_per_subject'], \
            "Total number of seconds with epochs and epoch len is exceeds the total amount of seconds, idiot!"

        # Set default to 19 channels
        self._num_channels = num_channels

    @property
    def name(self):
        return self._name

    @property
    def eeg_len(self):
        return self._eeg_len

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def eeg_info(self):
        return self._eeg_info

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

        # Verifies that all subjects in the various sets actually are present in the folder, else remove them from split
        train_subjects = verify_split_subjects(train_subjects, path=self.dataset_path)
        val_subjects = verify_split_subjects(val_subjects, path=self.dataset_path)
        test_subjects = verify_split_subjects(test_subjects, path=self.dataset_path)

        return train_subjects, val_subjects, test_subjects

    def load_targets(self, subjects: Tuple[str, ...], split,
                     get_stats=False) -> numpy.ndarray:  # type: ignore[type-arg, return]
        """
        Load target class labels for a given set of subjects based on the current task.

        This method handles different tasks by loading and possibly transforming class labels
        for the provided subjects. It supports tasks like 'CAUEEG-Abnormal benchmark' and
        'CAUEEG-Dementia benchmark', with different prediction types such as 'normal',
        'one_vs_all', and 'pairwise'. Depending on the task and prediction type, this method
        may apply one-hot encoding or binary transformation to the labels. It also optionally
        gathers label statistics if a split is provided.

        Parameters
        ----------
        subjects : Tuple[str, ...]
            A tuple of subject identifiers for which to load the target labels.
        split : str, optional
            The dataset split (e.g., 'train', 'test', 'validation') for which to load and
            possibly report label statistics. If not specified, statistics are not gathered.

        Returns
        -------
        numpy.ndarray
            An array of processed class labels for the given subjects. The processing depends
            on the current task and prediction type, including transformations like one-hot
            encoding or binary classification adjustment.

        Raises
        ------
        KeyError
            If the task name or class name for one-vs-all predictions is unrecognized.

        Side Effects
        ------------
        Calls `self.get_label_statistics` if a `split` is provided, to compute and log label
        statistics for the current dataset split.

        Notes
        -----
        This method relies on internal attributes such as `_task_name`, `_prediction_type`,
        and others to determine the appropriate processing of class labels for the subjects.
        """
        if split == "test":
            class_labels = np.array([self._merged_splits[sub]['class_label'] for sub in subjects])
        elif split == "val":
            class_labels = np.array([self._merged_splits[sub]['class_label'] for sub in subjects])
            class_labels = np.repeat(class_labels, self._num_val_epochs)
        else:
            # Repeat the classes num_epochs times....
            class_labels = np.array([self._merged_splits[sub]['class_label'] for sub in subjects])
            class_labels = np.repeat(class_labels, self._epochs)

        if self._task_name == 'CAUEEG-Abnormal benchmark':
            if get_stats:
                self.get_label_statistics(class_labels=class_labels, classes={"0": 'Normal', "1": 'Abnormal'},
                                          split=split)
            return class_labels
        elif self._task_name == 'CAUEEG-Dementia benchmark':
            class_labels: numpy.ndarray = torch.nn.functional.one_hot(  # type: ignore[type-arg]
                torch.from_numpy(class_labels), num_classes=len(self._class_name_to_label)).numpy()
            if get_stats:
                self.get_label_statistics(class_labels=class_labels,
                                          classes={"0": "Normal", "1": "MCI", "2": "Dementia"},
                                          split=split)

            return class_labels
        else:
            raise KeyError(f"Unrecognized task name: {self._task_name}")

    def load_eeg_data(self, subjects: Tuple[str, ...], split=None,
                      plot: bool = False) -> numpy.ndarray:  # type: ignore[type-arg]
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
        split: str
            whcich split, train, test or val
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
        >>> self.load_eeg_data(('subject1', 'subject2'), plot=True)
        # This will load the EEG data for 'subject1' and 'subject2', plot the raw data, and return the data array.
        """
        if split == "test":
            num_epochs = 1
        elif split == "val":
            num_epochs = self._num_val_epochs
        else:
            num_epochs = self._epochs
        use_epochs = False
        if num_epochs == 1 or num_epochs == 0:
            data = numpy.zeros(shape=(len(subjects), self._num_channels, self._eeg_len))
        else:
            use_epochs = True
            data = numpy.zeros(shape=(len(subjects) * num_epochs, self._num_channels, self._eeg_len))

        # Progress bar
        pbar = tqdm(total=len(subjects), desc="Loading", unit="subjects", colour='green')
        # Loop through subject list
        for i, sub in enumerate(subjects):
            # Create subject numpy path
            subject_eeg_data_path = f"{self._dataset_path}/{sub}.npy"
            # Load the subject numpy file
            npy_data = numpy.load(subject_eeg_data_path)

            # Normalize the eeg signal
            npy_data = self.__normalize_data(x=npy_data)

            # Plotting function
            if plot:
                raw = mne.io.RawArray(data=npy_data, info=self.get_eeg_info(), verbose=False)
                raw.plot(block=True)

            if use_epochs:
                raw = mne.io.RawArray(data=npy_data, info=self.get_eeg_info(), verbose=False)
                overlap = (self._eeg_len / 2) / raw.info['sfreq'] if self._overlapping_epochs else 0

                epochs = mne.make_fixed_length_epochs(raw=raw, duration=self._num_seconds, overlap=overlap,
                                                      preload=True, verbose=False)
                epoch_npy_data = epochs.get_data(copy=False)

                # Select the first epochs for test and val
                if split == "test":
                    npy_data = epoch_npy_data[:num_epochs, :, :]
                elif split == "val":
                    max_num_epochs, _, _ = epoch_npy_data.shape
                    # Spread out the epochs maximally
                    indices = np.linspace(0, max_num_epochs - 1, num_epochs + 1)
                    indices = np.round(indices).astype(int)
                    indices = np.unique(indices)[:-1]
                    npy_data = epoch_npy_data[indices, :, :]
                else:
                    max_num_epochs, _, _ = epoch_npy_data.shape

                    if max_num_epochs < num_epochs:
                        raise ValueError(f"Not enough epochs for subject: {sub}")

                    if self._epoch_structure == "all":
                        if max_num_epochs > num_epochs:
                            npy_data = epoch_npy_data[:num_epochs, :, :]
                        else:
                            npy_data = epoch_npy_data
                    else:
                        if self._epoch_structure == "random":
                            indices = np.random.choice(max_num_epochs, size=num_epochs, replace=False)
                        else:
                            indices = np.linspace(0, max_num_epochs - 1, num_epochs)
                            indices = np.round(indices).astype(int)
                            indices = np.unique(indices)

                        npy_data = epoch_npy_data[indices, :, :]

                cur_index = 0
                for j in range((i * num_epochs), (i * num_epochs) + num_epochs):
                    data[j] = npy_data[cur_index]
                    cur_index += 1
                pbar.update(1)
            else:
                if self._eeg_len > npy_data.shape[1]:
                    raise ValueError("Specified EEG length is longer than the actual recording.")

                npy_data = npy_data[:, 0: self._eeg_len]

                # Add the data to the empty numpy array
                data[i] = npy_data
                pbar.update(1)

        # Close the progress bare, this could probably be avoided with a with statement instead, but...
        pbar.close()
        return data

    def load_ages(self, subjects: Tuple[str, ...], split, add_noise=False, noise_level=0.1) -> numpy.ndarray:
        """ Load age of the subjects.

        This function receives a tuple of subject IDs, it loops through these subjects and extracts the age from
        the loaded dictionary.

        Parameters
        ----------
        split
        noise_level
        add_noise
        subjects: Tuple[str, ...]
            Subject IDs

        Returns
        -------
        data: np.ndarray
            structure = [60, 65, 70, ..., n_subjects], shape=(n_subjects, 1)
        """
        transformed_ages = self._ageScaler.transform(sub_ids=subjects, add_noise=add_noise, noise_level=noise_level)

        if split == "test":
            num_epochs = 1
        elif split == "val":
            num_epochs = self._num_val_epochs
        else:
            num_epochs = self._epochs
        return np.repeat(transformed_ages, num_epochs)

    @staticmethod
    def __normalize_data(x):

        x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-8)
        return x

    @staticmethod
    def _read_config(json_path: str):
        with open(json_path) as json_reader:
            return json.load(json_reader)

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
        if self._preprocessing_steps['downsample']:
            sfreq = self._preprocessing_steps['high_freq'] * self._preprocessing_steps['nyquist']
        else:
            sfreq = 200  # From the paper
        return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    def get_label_statistics(self, class_labels, classes, split):
        """
        Calculate and log statistics for given class labels within a dataset split.

        This static method computes the count and proportion of each class label
        in a given set of labels, based on a provided mapping of class identifiers to names.
        It logs these statistics using the provided split name for context. The results are
        printed to standard output and logged as parameters in MLflow with detailed statistics,
        including counts and percentages of each class.

        Parameters
        ----------
        class_labels : Iterable
            An iterable (e.g., list or array) of class labels for which statistics are to be computed.
        classes : dict
            A dictionary mapping class identifiers (as strings) to class names. This mapping is used
            to translate class labels into human-readable class names for reporting.
        split : str
            The name of the dataset split (e.g., 'train', 'validation', 'test') for which statistics
            are being calculated. This is used for labeling purposes in the output.

        Raises
        ------
        KeyError
            If an unrecognized class label is encountered, indicating a mismatch between the provided
            class labels and the class identifier-to-name mapping.

        Side Effects
        ------------
        - Prints class label statistics (count and proportion) to standard output, formatted by the
          dataset split and class names.
        - Logs the computed statistics to MLflow, categorized under a parameter named after the
          dataset split and the phrase "Class Statistics".
        """
        label_counts = {class_name: 0 for class_name in classes.values()}

        for c in class_labels:
            if c.shape[0] == 3:
                cl_lab = np.argmax(c)
            else:
                cl_lab = c

            if str(cl_lab) in classes:
                label_counts[classes[str(cl_lab)]] += 1
            else:
                raise KeyError(f"Unrecognized key: {cl_lab} in dict: {classes}")

        total_labels = len(class_labels)
        stats = {}
        for class_name, count in label_counts.items():
            proportion = (count / total_labels) * 100
            # print(f"{split.upper()} - {class_name.upper()}:\n\tCount: {count}\n\tPropo: {proportion:.2f}")
            stats[f"{class_name.upper()}"] = f" count: {count}, %: {proportion}"

        # Append the test to the file
        path = os.path.join(self._save_dir, "split_statistics.txt")

        with open(path, "a") as file:
            for key, value in stats.items():
                file.write(f"{split.upper()}\t --> {key}\t: {value}\n")
            file.write("\n")

    def get_class_weights(self, subjects: str, normalize: bool = False) -> torch.Tensor:
        """
        Calculate class weights for a given dataset split.

        This method calculates class weights for a given dataset split based on the class distribution
        within the split. The class weights are computed as the inverse of the class proportions in the split
        """
        class_labels = np.array([self._merged_splits[sub]['class_label'] for sub in subjects])
        class_weights = {}
        total_samples = len(class_labels)

        # Calculate the proportion of each class and then the inverse of that proportion as the class weight
        for class_label, count in zip(*np.unique(class_labels, return_counts=True)):
            proportion = count / total_samples
            class_weights[class_label] = (1 / proportion)

        if normalize:
            # Normalize class weights to sum to 1
            total_weight = sum(class_weights.values())
            for class_label in class_weights:
                class_weights[class_label] /= total_weight

        # Convert the dictionary of class weights to a list ordered by class label and then to a torch.Tensor
        sorted_weights = [class_weights[key] for key in sorted(class_weights.keys())]

        # Return the class weights as a torch.Tensor
        return torch.tensor(sorted_weights, dtype=torch.float)
