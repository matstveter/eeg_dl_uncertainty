import os
import pickle
from typing import Any, Dict, List, Union
import mne
import json

from eegDlUncertainty.experiments.dataset_shift_experiment import eval_dataset_shifts
from eegDlUncertainty.experiments.ood_experiments import ood_exp
from eegDlUncertainty.models.classifiers.ensemble import Ensemble


def create_ensemble_directory(run_path):
    """
    Create directories for ensemble models

    Parameters
    ----------
    run_path: str
        Path to the run directory

    Returns
    -------
    run_path_5: str
        Path to the ensemble_5 directory
    run_path_20: str
        Path to the ensemble_20 directory

    """
    run_path_5 = os.path.join(run_path, "ensemble_5")
    os.makedirs(run_path_5, exist_ok=True)
    run_path_20 = os.path.join(run_path, "ensemble_20")
    os.makedirs(run_path_20, exist_ok=True)

    return run_path_5, run_path_20


def read_json_file(json_file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """ Function that receives a json file and reads it and return

    :param json_file_path: Path to config file
    :return: dict
    """

    with open(json_file_path) as config_file:
        config = json.load(config_file)

    if not isinstance(config, (dict, list)):
        raise ValueError("JSON content must be a dictionary or a list")

    return config


def read_eeg_file(eeg_file_path: str):
    """
    Read EEG data from a file and return the raw data object.

    This function is specifically designed for reading EEG files in the EDF format. It leverages the
    MNE library to load the data. If the file is not in EDF format or an error occurs during reading,
    appropriate exceptions are raised or errors are printed.

    Parameters
    ----------
    eeg_file_path : str
        The file path to the EEG data file. The file should be in EDF format.

    Returns
    -------
    mne.io.Raw
        An object containing raw EEG data, as loaded by MNE's `read_raw_edf` function.

    Raises
    ------
    NotImplementedError
        If the file extension is not '.edf', this error is raised indicating that only EDF files are supported.
    ValueError
        If MNE's `read_raw_edf` function encounters a problem reading the EDF file, a ValueError is raised with a
        message indicating the issue.

    Examples
    --------
    >>> raw_data = read_eeg_file("path/to/eeg_file.edf")
    >>> print(raw_data)
    <RawEDF  |  file.edf, n_channels x n_times : 8 x 2380 (47.6 sec), ~153 kB, data loaded>

    Note
    ----
    The `read_raw_edf` function from MNE is used to read the EDF file, and it requires the `preload` parameter to be set
    to True for immediate data loading.
    """
    root, ext = os.path.splitext(eeg_file_path)
    if ext.lower() == ".edf":  # Ensure extension comparison is case-insensitive
        return mne.io.read_raw_edf(input_fname=eeg_file_path, preload=True, verbose=False,
                                   exclude=['EKG', 'Photic'])
    elif ext.lower() == ".set":
        return mne.io.read_raw_eeglab(input_fname=eeg_file_path, verbose=False, preload=True)
    else:
        raise NotImplementedError(f"Unsupported file type for {eeg_file_path}. Only EDF files are supported.")


def view_eeg_from_file_path(file_path: str):
    """
    Plot EEG data from a specified file.

    This function reads EEG data from a file and plots it using a plotting function associated with the `raw` object.
    The plotting is set with automatic scaling for better visualization and blocking mode to keep the plot window open.

    Parameters
    ----------
    file_path : str
        The path to the file containing EEG data. The file should be compatible with the `read_eeg_file` function.

    Notes
    -----
    - The function assumes that the `read_eeg_file` function is available and can read the specified EEG file format.
    - The 'scalings' parameter is set to 'auto' to automatically adjust the scale of the plots for better visibility.
    - The 'block' parameter is set to True, which means the plot will remain open until it is manually closed.

    """
    raw = read_eeg_file(eeg_file_path=file_path)
    raw.plot(scalings='auto', block=True)


# def save_dict_to_pickle(data_dict, path, name):
#     full_path = os.path.join(path, f"{name}.pkl")
#
#     # Check if file exists, if so, try another name
#     if os.path.exists(full_path):
#         print("File already exists, trying another name.")
#         i = 1
#         while os.path.exists(full_path):
#             full_path = os.path.join(path, f"{name}_{i}.pkl")
#             i += 1
#
#     with open(full_path, 'wb') as file:
#         pickle.dump(data_dict, file)


def run_ensemble_experiment(*, classifiers, device,
                            experiment_path, dataset, dataset_version, num_seconds,
                            age_scaling, batch_size, criterion,
                            test_subjects, val_loader, test_loader, save_name="ensemble_test_results"):

    # Initialize ensemble model with the trained classifiers
    ens = Ensemble(classifiers=classifiers, device=device)
    # Set the temperature scale for the ensemble
    ens.set_temperature_scale_ensemble(data_loader=val_loader, device=device, criterion=criterion,
                                       save_path=experiment_path)
    ens.ensemble_performance_and_uncertainty(data_loader=test_loader, device=device, save_path=experiment_path,
                                             save_to_mlflow=True, save_to_pickle=True,
                                             save_name=save_name)
    # Evaluate the dataset shifts on the ensemble model using the test set
    eval_dataset_shifts(ensemble_class=ens, test_subjects=test_subjects, dataset=dataset,
                        device=device, use_age=True, batch_size=batch_size,
                        save_path=experiment_path)
    # Evaluate the dataset shifts on the ensemble models, when all subjects ages are set to the same value (mean)
    eval_dataset_shifts(ensemble_class=ens, test_subjects=test_subjects, dataset=dataset,
                        device=device, use_age=False, batch_size=batch_size,
                        save_path=experiment_path)
    # Run the OOD experiment
    ood_exp(ensemble_class=ens, dataset_version=dataset_version,
            num_seconds=num_seconds,
            age_scaling=age_scaling, device=device, batch_size=batch_size,
            save_path=experiment_path, train_dataset=dataset)
