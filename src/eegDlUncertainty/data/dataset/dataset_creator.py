import multiprocessing
import os
import shutil
import time

import autoreject
from typing import Any, Dict, Tuple, Union

import mne
import numpy as np
import pandas as pd

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


def process_events(event_json_path: str, event_key: str) -> Tuple[Dict[str, Union[int, Dict[str, int]]], float]:
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
    durations = []
    for s in subjects:
        sub, _ = os.path.splitext(s)
        # Read json file
        events = read_json_file(json_file_path=os.path.join(event_json_path, s))

        subject_event[sub] = 0
        last_event_time = 0

        temp_dict = {'start_time': 0, 'end_time': 0}
        # Find the first occurrence of either eyes open or eyes closed, because we do not want the first events
        for event in events:
            start_time = event[0]
            event_name = event[1]

            # The last one is one specif case...
            if (event_name.lower() == "eyes open" or event_name.lower() == "eyes closed" or
                    event_name == "passive eye open"):
                temp_dict['start_time'] = int(start_time)
                break

        # Find the max lenght of the recording, based on either the occurrence of a photic event or just the last event
        for event in events:
            start_time = event[0]
            event_name = event[1]
            last_event_time = int(start_time)

            if "photic" in event_name.lower():
                temp_dict['end_time'] = int(start_time)
                break

        # If no photic event is found, set to the last event time
        if temp_dict['end_time'] == 0:
            temp_dict['end_time'] = last_event_time

        durations.append((temp_dict['end_time'] - temp_dict['start_time']) / 200)
        subject_event[sub] = temp_dict
    return subject_event, min(durations)


def process_eeg_data(config: Dict[str, Any], conf_path: str) -> None:
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
        events, min_duration = process_events(event_json_path=config['file_paths']['eeg_events'],
                                              event_key='Eyes Closed')
    else:
        events = None

    # Check if preprocess exists in the config file
    if "preprocessing" in config:
        preprocess = config['preprocessing']
    else:
        raise ValueError("Missing preprocessing entry in the config file!")

    outp_path = create_output_folder(base_path=config['file_paths']['output_directory'])

    start = time.perf_counter()
    if config['use_multiprocessing']:
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 3)) as pool:
            pool.starmap(process_eeg_file,
                         [(sub_eeg, eeg_path, events, preprocess, preprocess['nyquist'], outp_path,
                           preprocess['autoreject'])
                          for sub_eeg in eeg_files])
    else:
        for sub_eeg in eeg_files:
            process_eeg_file(sub_eeg=sub_eeg, eeg_path=eeg_path, events=events, preprocess=preprocess,
                             nyquist=preprocess['nyquist'], out_p=outp_path, use_autoreject=preprocess['autoreject'])
    print(f"Processing finished, time used: {time.perf_counter() - start}")
    shutil.copy(src=conf_path, dst=outp_path)


def process_eeg_file(sub_eeg, eeg_path, events, preprocess, nyquist, out_p, use_autoreject):
    """
    Process a single EEG file by applying specified preprocessing steps.

    This function reads an EEG file, optionally adjusts the start time based on event information,
    crops the data to a specified duration, applies bandpass filtering, and optionally down-samples the data.
    The processed EEG data is then saved to a specified output path.

    Parameters
    ----------
    use_autoreject
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
        The Nyquist frequency used as a multiplier to determine the new sampling rate when down-sampling.
    out_p : str
        The output directory where the processed EEG data files will be saved.

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
    - Optionally down-sampling the data if `preprocess['downsample']` is True, using a new sampling rate determined
      by `preprocess['high_freq'] * nyquist`.

    Examples
    --------
    """
    raw_eeg_data = read_eeg_file(eeg_file_path=os.path.join(eeg_path, str(sub_eeg)))
    if raw_eeg_data is None:
        print("Raw file is None, returning")
        return
    subject_id, _ = os.path.splitext(sub_eeg)

    start_time = events[subject_id]['start_time'] / raw_eeg_data.info['sfreq']
    if preprocess['use_end_time']:
        end_time = events[subject_id]['end_time'] / raw_eeg_data.info['sfreq']
    else:
        end_time = start_time + preprocess['num_seconds_per_subject']

    # Check if the end_time is higher than the recording somehow...
    if end_time > raw_eeg_data.times[-1]:
        end_time = raw_eeg_data.times[-1]

    total_time = end_time - start_time

    if total_time < preprocess['num_seconds_per_subject']:
        print("Data is too short, skipping")
        return

    raw_eeg_data.crop(tmin=start_time, tmax=end_time, verbose=False, include_tmax=False)

    # Lowpass and high_pass filter the data
    if preprocess['use_notch']:
        raw_eeg_data.notch_filter(freqs=50, verbose=False)

    raw_eeg_data.filter(l_freq=preprocess['low_freq'], h_freq=preprocess['high_freq'], verbose=False)

    new_sampling_rate = raw_eeg_data.info['sfreq']
    # If downsample is set to true, use the nyquist argument to specify the new sampling rate
    if preprocess['downsample']:
        if preprocess['high_freq'] * nyquist > 200:
            print(f"WARNING: Original sampling frequency is 200, with current config, "
                  f"sampling is done to {preprocess['high_freq'] * nyquist}")
        new_sampling_rate = preprocess['high_freq'] * nyquist  # Defaults to 3, as 2 is the absolute min
        raw_eeg_data.resample(new_sampling_rate, verbose=False)

    if preprocess['rereference_avg']:
        raw_eeg_data.set_eeg_reference('average', verbose=False)

    if use_autoreject:
        # Your channel names
        ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4',
                    'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6',
                    'Fz', 'Cz', 'Pz']
        # Ensure channel count matches
        if len(raw_eeg_data.ch_names) != len(ch_names):
            raise ValueError("Mismatch between provided and actual channel names count.")

        # Rename channels and apply standard 10-20 montage
        raw_eeg_data.rename_channels(dict(zip(raw_eeg_data.ch_names, ch_names)))
        raw_eeg_data.set_montage(mne.channels.make_standard_montage('standard_1020'))

        epochs = mne.make_fixed_length_epochs(raw_eeg_data.copy(), duration=preprocess['autoreject_epochs'],
                                              preload=True, verbose=False)

        ar = autoreject.AutoReject(random_state=11,
                                   n_jobs=1, verbose=False, cv=min(10, len(epochs)))
        ar.fit(epochs)
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        if len(epochs_ar) == 0:
            print(f"Skipping {subject_id} due to no epochs")
            return
        # Concatenate epochs, this can cause discontinuities in the data, but it will be handled by reading the same
        # of seconds in the
        data = np.concatenate(epochs_ar.get_data(copy=True), axis=1)

        # Check if the data is too short
        remaining_time = data.shape[1] / new_sampling_rate
        if remaining_time < preprocess['num_seconds_per_subject']:
            print(f"Data is too short, skipping {subject_id}")
            return
    else:
        data = raw_eeg_data.get_data()

    save_path = os.path.join(out_p, f"{subject_id}.npy")
    np.save(save_path, data)
    print(f"{subject_id}: Done")


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
        if "tdbrain" in conf_path:
            create_tdbrain_dataset(config=config, conf_path=conf_path)
        elif "greek" in conf_path:
            create_greek_dataset(config=config, conf_path=conf_path)
        elif "mpi" in conf_path:
            create_MPI_dataset(config=config, conf_path=conf_path)
        else:
            # Now it's safe to pass config as it's confirmed to be a Dict
            process_eeg_data(config=config, conf_path=conf_path)
            # process_eeg_data_autoreject(config=config)
    else:
        # Handle the case where config is not a Dict, maybe raise an error or log a warning
        raise TypeError("Expected config to be a dictionary")


def get_cau_eeg_channels():
    return ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4',
            'P4', 'O2', 'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz']


def create_tdbrain_dataset(config, conf_path):
    """ This function is used to generate EEGs with the same preprocessing steps as CAUEEG.

    TDbrain article: https://www.nature.com/articles/s41597-022-01409-z

    Parameters
    ----------
    config: dict
        preprocessing steps, paths etc.
    conf_path: path to config file

    Returns
    -------

    """
    if "eeg_data" in config['file_paths']:
        eeg_path = config['file_paths']['eeg_data']
    else:
        raise ValueError("Please specify the path to the EEG files in the config file in the script folder. "
                         "\nkey: 'eeg_data': path")

    data_info = pd.read_csv(os.path.join(eeg_path, "TDBRAIN_participants_V2.tsv"), sep='\t')

    if config['use_derivatives']:
        eeg_path = os.path.join(eeg_path, "derivatives")

    output_path = config['file_paths']['output_directory']
    outp_path = create_output_folder(base_path=output_path)

    filtered_data = data_info[data_info['indication'] == "HEALTHY"]

    desired_channels = get_cau_eeg_channels()

    ext = "ses-1/eeg/"

    rename_dict = {
        'T7': 'T3',
        'T8': 'T4',
        'P7': 'T5',
        'P8': 'T6'
    }
    preprocess = config['preprocessing']

    for sub in filtered_data['participants_ID']:
        if config['use_derivatives']:
            eeg_file = os.path.join(eeg_path, sub, ext, f"{sub}_ses-1_task-restEC_eeg.csv")
            eeg_df = pd.read_csv(eeg_file)
            ch_names = eeg_df.columns.tolist()
            ch_to_remove = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']
            eeg_df.drop(columns=ch_to_remove, inplace=True)
            # Convert to microvolts
            data = eeg_df.to_numpy().T / 1e6
            sfreq = 500
            info = mne.create_info(ch_names=eeg_df.columns.tolist(), sfreq=sfreq, ch_types='eeg', verbose=False)
            raw = mne.io.RawArray(data, info, verbose=False)
        else:
            eeg_file = os.path.join(eeg_path, sub, ext, f"{sub}_ses-1_task-restEC_eeg.vhdr")
            raw = mne.io.read_raw_brainvision(vhdr_fname=eeg_file, preload=True, verbose=False)

        raw.rename_channels(rename_dict, verbose=False)
        raw.pick(desired_channels, verbose=False)

        minimum_end_time = preprocess['start'] + preprocess['num_seconds_per_subject']

        # Align available end_time with epochs
        epoch_length = preprocess['autoreject_epochs']
        end_time = int(raw.times[-1] / epoch_length) * epoch_length

        # Check if end_time is sufficient
        if end_time < minimum_end_time:
            print(f"Data is too short ({end_time}s), skipping {sub}")
            continue

        # Crop to the aligned epoch boundary
        raw.crop(tmin=preprocess['start'],
                 tmax=end_time,
                 verbose=False, include_tmax=False)

        # plot the power spectral density
        if preprocess['use_notch']:
            raw.notch_filter(freqs=50, verbose=False)
            print("Applying notch filter")

        # Lowpass and high_pass filter the data
        raw.filter(l_freq=preprocess['low_freq'], h_freq=preprocess['high_freq'], verbose=False)
        raw.resample(preprocess['sfreq'], verbose=False)

        raw.set_eeg_reference('average', verbose=False)

        if preprocess['autoreject']:
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
            epochs = mne.make_fixed_length_epochs(raw.copy(), duration=preprocess['autoreject_epochs'],
                                                  preload=True, verbose=False)
            if len(epochs) <= 2:
                print(f"Skipping {sub} due to no epochs: {len(epochs)}")
                continue
            ar = autoreject.AutoReject(random_state=11,
                                       n_jobs=1, verbose=False, cv=min(10, len(epochs)))
            ar.fit(epochs)
            epochs_ar, reject_log = ar.transform(epochs, return_log=True)
            if len(epochs_ar) == 0:
                print(f"Skipping {sub} due to no epochs")
                continue
            # Concatenate epochs, this can cause discontinuities in the data, but it will be handled by reading the same
            # of seconds in the
            data = np.concatenate(epochs_ar.get_data(copy=True), axis=1)

            # Check if the data is too short
            remaining_time = data.shape[1] / preprocess['sfreq']

            print(remaining_time)

            if remaining_time < preprocess['num_seconds_per_subject']:
                print(f"Data is too short, skipping {sub}")
                continue
        else:
            data = raw.get_data()

        save_path = os.path.join(outp_path, f"{sub}.npy")
        np.save(save_path, data)
        print(f"{sub}: Done")

    shutil.copy(src=conf_path, dst=outp_path)


def create_greek_dataset(config, conf_path):
    if "eeg_data" in config['file_paths']:
        eeg_path = config['file_paths']['eeg_data']
    else:
        raise ValueError("Please specify the path to the EEG files in the config file in the script folder. "
                         "\nkey: 'eeg_data': path")

    data_info = pd.read_csv(os.path.join(eeg_path, "participants.tsv"), sep='\t').to_dict()['participant_id']

    if config['use_derivatives']:
        eeg_path = os.path.join(eeg_path, "derivatives")
        output_path = os.path.join(config['file_paths']['output_directory'], "derivatives")
    else:
        output_path = config['file_paths']['output_directory']

    outp_path = create_output_folder(base_path=output_path)
    desired_channels = get_cau_eeg_channels()
    preprocess = config['preprocessing']

    for sub in data_info.values():
        file_path = os.path.join(eeg_path, sub, "eeg", f"{sub}_task-eyesclosed_eeg.set")
        raw = mne.io.read_raw_eeglab(input_fname=file_path, verbose=False, preload=True)

        raw.pick(desired_channels, verbose=False)

        minimum_end_time = preprocess['start'] + preprocess['num_seconds_per_subject']

        # Align available end_time with epochs
        epoch_length = preprocess['autoreject_epochs']
        end_time = int(raw.times[-1] / epoch_length) * epoch_length

        # Check if end_time is sufficient
        if end_time < minimum_end_time:
            print(f"Data is too short ({end_time}s), skipping {sub}")
            continue

        # Crop to the aligned epoch boundary
        raw.crop(tmin=preprocess['start'],
                 tmax=end_time,
                 verbose=False, include_tmax=False)

        if preprocess['use_notch'] and not config['use_derivatives']:
            # plot the power spectral density
            raw.notch_filter(freqs=50, verbose=False)
            raw.notch_filter(freqs=62.5, verbose=False)
            print("Applying notch filter")

        # Lowpass and high_pass filter the data
        raw.filter(l_freq=preprocess['low_freq'], h_freq=preprocess['high_freq'], verbose=False)
        raw.resample(preprocess['sfreq'], verbose=False)

        raw.set_eeg_reference('average', verbose=False)

        if preprocess['autoreject']:
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
            epochs = mne.make_fixed_length_epochs(raw.copy(), duration=preprocess['autoreject_epochs'],
                                                  preload=True, verbose=False)

            if len(epochs) <= 2:
                print(f"Skipping {sub} due to no epochs")
                continue

            ar = autoreject.AutoReject(random_state=11,
                                       n_jobs=1, verbose=False, cv=min(10, len(epochs)))
            ar.fit(epochs)
            epochs_ar, reject_log = ar.transform(epochs, return_log=True)
            if len(epochs_ar) == 0:
                print(f"Skipping {sub} due to no epochs")
                continue
            # Concatenate epochs, this can cause discontinuities in the data, but it will be handled by reading the same
            # of seconds in the
            data = np.concatenate(epochs_ar.get_data(copy=True), axis=1)

            # Check if the data is too short
            remaining_time = data.shape[1] / preprocess['sfreq']

            print(remaining_time)

            if remaining_time < preprocess['num_seconds_per_subject']:
                print(f"Data is too short, skipping {sub}")
                continue
        else:
            data = raw.get_data()

        save_path = os.path.join(outp_path, f"{sub}.npy")
        np.save(save_path, data)
        print(f"{sub}: Done")

    shutil.copy(src=conf_path, dst=outp_path)


def create_MPI_dataset(config, conf_path):
    if "eeg_data" in config['file_paths']:
        eeg_path = config['file_paths']['eeg_data']
    else:
        raise ValueError("Please specify the path to the EEG files in the config file in the script folder. "
                         "\nkey: 'eeg_data': path")

    data_info = pd.read_csv(os.path.join(eeg_path, "participants.csv"), sep=',').to_dict()['ID']
    outp_path = create_output_folder(base_path=config['file_paths']['output_directory'])
    desired_channels = get_cau_eeg_channels()
    rename_dict = {
        'T7': 'T3',
        'T8': 'T4',
        'P7': 'T5',
        'P8': 'T6'
    }
    preprocess = config['preprocessing']

    all_files = os.listdir(eeg_path)

    for sub in data_info.values():
        if sub in all_files:
            eeg_file = os.path.join(eeg_path, sub, f"{sub}.set")
            raw = mne.io.read_raw_eeglab(input_fname=eeg_file, preload=True, verbose=False)
            try:
                raw.rename_channels(rename_dict, verbose=False)
                raw.pick(desired_channels, verbose=False)
            except ValueError:
                print(f"Skipping subject: {sub}")
                continue

            minimum_end_time = preprocess['start'] + preprocess['num_seconds_per_subject']

            # Align available end_time with epochs
            epoch_length = preprocess['autoreject_epochs']
            end_time = int(raw.times[-1] / epoch_length) * epoch_length

            # Check if end_time is sufficient
            if end_time < minimum_end_time:
                print(f"Data is too short ({end_time}s), skipping {sub}")
                continue

            # Crop to the aligned epoch boundary
            raw.crop(tmin=preprocess['start'],
                     tmax=end_time,
                     verbose=False, include_tmax=False)

            if preprocess['use_notch']:
                try:
                    raw.notch_filter(freqs=50, verbose=False)
                except ValueError:
                    print(f"Skipping subject: {sub}")
                    continue

            # Lowpass and high_pass filter the data, using butterworth filter
            raw.filter(l_freq=preprocess['low_freq'], h_freq=preprocess['high_freq'], verbose=False, )
            raw.resample(preprocess['sfreq'], verbose=False)

            raw.set_eeg_reference('average', verbose=False)

            if preprocess['autoreject']:
                raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
                epochs = mne.make_fixed_length_epochs(raw.copy(), duration=preprocess['autoreject_epochs'],
                                                      preload=True, verbose=False)

                if len(epochs) <= 2:
                    print(f"Skipping {sub} due to no epochs")
                    continue

                ar = autoreject.AutoReject(random_state=11,
                                           n_jobs=1, verbose=False, cv=min(10, len(epochs)))
                ar.fit(epochs)
                epochs_ar, reject_log = ar.transform(epochs, return_log=True)
                if len(epochs_ar) == 0:
                    print(f"Skipping {sub} due to no epochs")
                    continue
                data = np.concatenate(epochs_ar.get_data(copy=True), axis=1)

                # Check if the data is too short
                remaining_time = data.shape[1] / preprocess['sfreq']

                if remaining_time < preprocess['num_seconds_per_subject']:
                    print(f"Data is too short, skipping {sub}")
                    continue
            else:
                data = raw.get_data()

            save_path = os.path.join(outp_path, f"{sub}.npy")
            np.save(save_path, data)
            print(f"Finished: {sub}")
        else:
            print(f"Missing participant: {sub}")
    shutil.copy(src=conf_path, dst=outp_path)
