import os
import mne
import numpy as np

from eegDlUncertainty.data.utils import read_eeg_file, read_json_file
from autoreject import AutoReject


def process_eeg_data_autoreject(config):
    file_paths = config['file_paths']
    preprocessing_steps = config['preprocessing']

    eeg_path = file_paths['eeg_data']
    events = file_paths['eeg_events']
    eeg_files = sorted(os.listdir(eeg_path))
    event_files = sorted(os.listdir(events))

    if config['use_multiprocessing'] and False:
        pass
    else:
        for sub_eeg in eeg_files:
            process_eeg_file(
                sub_eeg=sub_eeg,
                event_path=events,
                eeg_path=eeg_path,
                preprocess=preprocessing_steps,
                event_files=event_files
            )


def process_eeg_file(sub_eeg, event_path, eeg_path, preprocess, event_files):
    sub_id = sub_eeg.split(sep='.')[0]
    sub_event_id = sub_id + ".json"

    if sub_event_id not in event_files:
        print(f"Missing event file for subject: {sub_id}")
        return

    raw_eeg_data = read_eeg_file(eeg_file_path=os.path.join(eeg_path, str(sub_eeg)))
    if raw_eeg_data is None:
        print("Raw file is None, returning")
        return

    fix_montage(raw=raw_eeg_data)

    event_array, event_id, durations = calculate_events(path=os.path.join(event_path, sub_event_id),
                                                        sfreq=raw_eeg_data.info['sfreq'])
    tmin, tmax = 0, 20

    epochs = mne.Epochs(raw=raw_eeg_data, events=event_array, event_id=event_id,
                        tmin=tmin, tmax=tmax, preload=True, event_repeated="merge", baseline=(0, 0), verbose=False)
    eyes_closed_epochs = epochs['Eyes Closed']
    print(eyes_closed_epochs)
    print(durations, len(durations))
    n = len(durations) if len(durations) < 10 else 10
    ar = AutoReject(random_state=42, verbose=False, cv=n)
    eyes_closed_epochs_clean = ar.fit_transform(eyes_closed_epochs)
    print(eyes_closed_epochs_clean)


def calculate_events(path, sfreq, event_of_interest="Eyes Closed"):
    events = read_json_file(json_file_path=path)

    check_time_to_next = False
    duration_of_events = []
    for ev in events:
        time = int(ev[0])
        event_name = ev[1]

        if event_name == event_of_interest:
            check_time_to_next = True
            prev_time = time
            continue

        if check_time_to_next:
            duration_of_events.append((time - prev_time) / sfreq)
            check_time_to_next = False
            prev_time = 0
    event_id = {
        event_of_interest: 1,
        'Other': 2
    }

    # Convert events to the required format [onset, 0, event_id]
    event_array = []
    for e in events:
        onset = e[0]
        event_name = e[1]
        event_type = event_id[event_of_interest] if event_name == event_of_interest else event_id['Other']
        event_array.append([onset, 0, event_type])

    event_array = np.array(event_array, dtype=int)

    return event_array, event_id, duration_of_events


def fix_montage(raw):
    new_channel_names = {
        'Fp1-AVG': 'Fp1', 'F3-AVG': 'F3', 'C3-AVG': 'C3', 'P3-AVG': 'P3', 'O1-AVG': 'O1',
        'Fp2-AVG': 'Fp2', 'F4-AVG': 'F4', 'C4-AVG': 'C4', 'P4-AVG': 'P4', 'O2-AVG': 'O2',
        'F7-AVG': 'F7', 'T3-AVG': 'T7', 'T5-AVG': 'P7', 'F8-AVG': 'F8', 'T4-AVG': 'T8',
        'T6-AVG': 'P8', 'FZ-AVG': 'Fz', 'CZ-AVG': 'Cz', 'PZ-AVG': 'Pz'
    }
    raw.rename_channels(new_channel_names)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
