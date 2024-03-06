import os
from typing import Any, Dict, List, Union

import mne
import json


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
