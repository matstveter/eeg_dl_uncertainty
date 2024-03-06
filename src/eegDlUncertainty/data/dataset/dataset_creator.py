import math
import multiprocessing
import os
import shutil
from typing import Any, Dict
import numpy as np

from eegDlUncertainty.data.utils import read_eeg_file, read_json_file


def create_output_folder(base_path: str, base_name: str = "caug_numpy") -> str:
    """
    Creates a new folder with an incremented version number if the folder already exists.

    :param base_path: The base path where the folder will be created.
    :param base_name: The base name for the folder. Default is 'caug_numpy'.
    :return: The path of the created/new folder.
    """
    version = 1
    folder_name = f"{base_name}_v{version}"
    folder_path = os.path.join(base_path, folder_name)

    # Increment the version number until a unique folder name is found
    while os.path.exists(folder_path):
        version += 1
        folder_name = f"{base_name}_v{version}"
        folder_path = os.path.join(base_path, folder_name)

    # Create the folder with the final name
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def process_events(event_json_path: str, event_key: str) -> Dict[str, int]:
    """
    Process event JSON files in a given directory and extract the start time of a specified event.

    This function iterates through all JSON files in the specified directory, extracting the start
    time of the first occurrence of a specified event in each file. The result is a dictionary
    mapping the subject names (derived from the file names) to the start time of the event.

    Parameters
    ----------
    event_json_path : str
        The file path to the directory containing the event JSON files.
    event_key : str
        The event name to search for in each JSON file.

    Returns
    -------
    Dict[str, int]
        A dictionary mapping each subject (file name without extension) to the start time of the
        specified event. If the event is not found in a file, the subject is not included in the
        dictionary.

    Notes
    -----
    - Each JSON file in the directory is expected to contain a list of events, where each event
      is represented as a list with the first element being the start time (int) and the second
      element being the event name (str).
    - The function assumes that the JSON files are correctly formatted and will raise an error
      if unable to read or parse a file.
    - If multiple occurrences of the specified event exist in a file, only the first occurrence
      is considered.
    """
    # Get all the subject json file names
    subjects = sorted(os.listdir(event_json_path))

    # Empty dict for saving the events to each of the subject
    subject_event = {}

    # Loop through the subjects
    for s in subjects:
        sub, _ = os.path.splitext(s)
        # Read json file
        events = read_json_file(json_file_path=os.path.join(event_json_path, s))

        # Loop through the event list
        for event in events:
            # Extract the start time and the event name
            start_time = event[0]
            event_name = event[1]

            # If the first one is found, break out of the inner for loop
            if event_name == event_key:
                subject_event[sub] = int(start_time)
                break

    return subject_event


def process_eeg_data(config: Dict[str, Any], conf_path: str, nyquist: int = 3) -> None:
    """
    Process EEG (Electroencephalography) data files based on a given configuration.

    The function checks the provided configuration for the necessary paths to the EEG data files,
    event information, and preprocessing specifications. It processes each EEG file in parallel
    using multiprocessing, applying specified preprocessing steps and handling event information
    if available. The processed data is saved to an output directory, which is created if it doesn't exist.
    The configuration file is also copied to the output directory for record-keeping.

    Parameters
    ----------
    config : Dict[str, Any]
        A dictionary containing the configuration settings, including file paths for EEG data,
        event information, and preprocessing details.
    conf_path : str
        The path to the configuration file.
    nyquist : int, optional
        The Nyquist frequency to be considered for filtering the data, default is 3 Hz.

    Raises
    ------
    ValueError
        If the 'eeg_data' key is missing from the config file's file_paths section, or
        if the specified EEG file path is None or empty.
        If the 'preprocessing' key is missing from the config.

    Notes
    -----
    The function requires the configuration to have specific keys:
    - 'file_paths' with 'eeg_data', 'eeg_events', and 'output_directory' sub-keys.
    - 'preprocessing' detailing the preprocessing steps to be applied to the EEG data.

    It's assumed that the 'process_events' and 'process_eeg_file' functions are defined
    elsewhere and are responsible for processing event information and individual EEG files,
    respectively.

    Examples
    --------
    >>> config = {
    ...     "file_paths": {
    ...         "eeg_data": "/path/to/eeg/data",
    ...         "eeg_events": "/path/to/events/file",
    ...         "output_directory": "/path/to/output"
    ...     },
    ...     "preprocessing": {"filter": "bandpass", "freqs": [1, 30]}
    ... }
    >>> conf_path = "/path/to/config/file"
    >>> process_eeg_data(config, conf_path)
    """
    # Check that it is an entry in the config which specifies the path
    if "eeg_data" in config['file_paths']:
        eeg_path = config['file_paths']['eeg_data']

        if eeg_path is not None or eeg_path != "":
            eeg_files = sorted(os.listdir(eeg_path))
        else:
            raise ValueError("EEG file path is None or empty, please specify!")
    else:
        raise ValueError("Please specify the path to the EEG files in the config file in the script folder. "
                         "\nkey: 'eeg_data': path")

    # Check if there is an entry called eeg_events in the config file, if it is use the process-events function
    if "eeg_events" in config['file_paths']:
        events = process_events(event_json_path=config['file_paths']['eeg_events'], event_key='Eyes Open')
    else:
        events = None

    # Check if preprocess exists in the config file
    if "preprocessing" in config:
        preprocess = config['preprocessing']
    else:
        raise ValueError("Missing preprocessing entry in the config file!")

    outp_path = create_output_folder(base_path=config['file_paths']['output_directory'])

    with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as pool:
        pool.starmap(process_eeg_file,
                     [(sub_eeg, eeg_path, events, preprocess, nyquist, outp_path, preprocess['start'])
                      for sub_eeg in eeg_files])
    shutil.copy(src=conf_path, dst=outp_path)


def process_eeg_file(sub_eeg, eeg_path, events, preprocess, nyquist, out_p, start):
    """
    Process a single EEG file by applying specified preprocessing steps.

    This function reads an EEG file, optionally adjusts the start time based on event information,
    crops the data to a specified duration, applies bandpass filtering, and optionally downsamples the data.
    The processed EEG data is then saved to a specified output path.

    Parameters
    ----------
    sub_eeg : str
        The filename of the EEG data file to be processed.
    eeg_path : str
        The directory path containing the EEG data files.
    events : dict, optional
        A dictionary mapping subject IDs to event times. If provided, it is used to adjust the start time
        for processing the EEG data for each subject. If a subject ID is not found in the events dictionary,
        the function falls back to the default start time provided in the `start` parameter.
    preprocess : dict
        A dictionary containing preprocessing parameters including the number of seconds to process per subject,
        low and high frequency bounds for bandpass filtering, and whether to downsample the data.
    nyquist : int
        The Nyquist frequency used as a multiplier to determine the new sampling rate when downsampling.
    out_p : str
        The output directory where the processed EEG data files will be saved.
    start : int
        The default start time (in seconds) from which to begin processing the EEG data if no event information
        is provided or applicable for a subject.

    Notes
    -----
    The EEG data is expected to be in a format readable by `read_eeg_file`, which should return an object with
    methods for cropping, filtering, resampling, and accessing the data. The exact format and capabilities of
    this object are assumed to be compatible with MNE-Python's Raw objects, but the specific implementation may vary.

    The `events` parameter is optional. If not provided or if a subject's ID is not in the events dictionary,
    the function uses the default start time provided by the `start` parameter.

    The preprocessing steps include:
    - Cropping the data starting from either the event time or the default start time, for a duration specified by
      `preprocess['num_seconds_per_subject']`.
    - Applying a bandpass filter with low and high frequency bounds specified by `preprocess['low_freq']` and
      `preprocess['high_freq']`.
    - Optionally downsampling the data if `preprocess['downsample']` is True, using a new sampling rate determined
      by `preprocess['high_freq'] * nyquist`.

    Examples
    --------
    >>> preprocess = {
    ...     "num_seconds_per_subject": 300,
    ...     "low_freq": 0.1,
    ...     "high_freq": 40.0,
    ...     "downsample": True
    ... }
    >>> process_eeg_file("subject1.edf", "/data/eeg/", {"subject1": 100}, preprocess, 3, "/output/", 60)
    """
    raw_eeg_data = read_eeg_file(eeg_file_path=os.path.join(eeg_path, str(sub_eeg)))
    subject_id, _ = os.path.splitext(sub_eeg)

    # Use information from the event or use from the config file
    if events is not None:
        try:
            start = math.ceil(events[subject_id] / raw_eeg_data.info['sfreq'])
        except KeyError:
            print("Subject ID missing event_key, setting the start to start from config!")
            start = start
    else:
        # Starting at a time point
        start = start

    end_time = preprocess['num_seconds_per_subject'] + start

    time_max = raw_eeg_data.times[-1]
    # Check that the recording is shorter or equal the end time
    if time_max >= end_time:
        raw_eeg_data.crop(tmin=start, tmax=end_time, verbose=False, include_tmax=False)
    # Check if we can start the recording from an earlier position...
    elif time_max >= preprocess['num_seconds_per_subject']:
        start = time_max - preprocess['num_seconds_per_subject']
        # Check if we  start earlier in the recording
        raw_eeg_data.crop(tmin=start, tmax=time_max, verbose=False, include_tmax=False)
    else:
        print(f"Subject {sub_eeg} data is too short, skipping...")
        return

    # Lowpass and high_pass filter the data
    raw_eeg_data.filter(l_freq=preprocess['low_freq'], h_freq=preprocess['high_freq'], verbose=False)

    # If downsample is set to true, use the nyquist argument to specify the new sampling rate
    if preprocess['downsample']:
        new_sampling_rate = preprocess['high_freq'] * nyquist  # Defaults to 3, as 2 is the absolute min
        raw_eeg_data.resample(new_sampling_rate, verbose=False)

    data = raw_eeg_data.get_data()

    save_path = os.path.join(out_p, f"{subject_id}.npy")
    np.save(save_path, data)


def create_eeg_dataset(conf_path: str):
    """
    Create an EEG dataset based on configuration settings provided in a JSON file.

    This function reads a JSON configuration file, verifies that it is a valid dictionary,
    and then proceeds to process EEG data according to the settings specified in the configuration.
    If the configuration file does not exist or is not in the expected format, appropriate errors
    are raised.

    Parameters
    ----------
    conf_path : str
        The file path to the JSON configuration file. The configuration should specify all
        necessary settings for EEG data processing.

    Raises
    ------
    FileNotFoundError
        If the specified JSON configuration file does not exist at the given path.
    TypeError
        If the content of the JSON configuration file is not a dictionary, indicating invalid
        or corrupt configuration data.

    Notes
    -----
    - The JSON configuration file should adhere to a predefined schema expected by `process_eeg_data`,
      including necessary parameters such as data paths, preprocessing steps, and analysis parameters.
    - This function delegates the actual EEG data processing to another function (`process_eeg_data`),
      which is not detailed here.

    Examples
    --------
    Suppose you have a configuration JSON file named `eeg_config.json` in the current directory,
    you can create an EEG dataset like this:

    >>> create_eeg_dataset('eeg_config.json')
    """
    # Check if file exists, load file, or raise FileNotFoundError
    if os.path.isfile(conf_path):
        config = read_json_file(json_file_path=conf_path)
    else:
        raise FileNotFoundError(f"Could not find json file '{conf_path}' in folder '{os.path.dirname(__file__)}'!")

    if isinstance(config, dict):
        # Now it's safe to pass config as it's confirmed to be a Dict
        process_eeg_data(config=config, conf_path=conf_path)
    else:
        # Handle the case where config is not a Dict, maybe raise an error or log a warning
        raise TypeError("Expected config to be a dictionary")
