import json
from typing import List, Optional, Tuple
import os


def check_folder(path, path_ext="figures"):
    full_path = os.path.join(path, path_ext)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path


def get_baseparameters_from_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)

    # Retrieve parameters with type hints
    prediction: str = config.get('data', {}).get('prediction')
    dataset_version: str = config.get('data', {}).get('version')
    eeg_epochs: str = config.get('data', {}).get('eeg_epochs')
    epoch_overlap: int = config.get('data', {}).get('overlapping_epochs')
    num_seconds: int = config.get('data', {}).get('num_seconds')
    use_age: bool = config.get('data', {}).get('use_age')
    age_scaling: str = config.get('data', {}).get('age_scaling')
    classifier_name: str = str(config.get('model', {}).get('name', ''))
    learning_rate: float = float(config.get('hyperparameters', {}).get('learning_rate'))
    batch_size: int = config.get('hyperparameters', {}).get('batch_size')
    augmentations: Optional[List[str]] = config.get('hyperparameters', {}).get('augmentations')
    training_epochs: int = config.get('model', {}).get('epochs')
    earlystopping: int = config.get('model', {}).get('earlystopping')
    mc_dropout_enabled: bool = config.get('mc_dropout', {}).get('enabled', False)
    mc_dropout_rate: float = config.get('mc_dropout', {}).get('dropout_rate')
    swa_enabled: bool = config.get('swa', {}).get('enabled', False)
    swa_lr: float = config.get('swa', {}).get('swa_lr')
    swa_epochs: int = config.get('swa', {}).get('swa_epochs')
    swag_enabled: bool = config.get('swag', {}).get('enabled', False)
    swag_lr: float = config.get('swag', {}).get('swag_lr')
    swag_freq: int = config.get('swag', {}).get('swag_freq')

    possible_eeg_epochs = ['all', 'spread', 'random']

    if eeg_epochs not in possible_eeg_epochs:
        raise KeyError(f"EEG epochs should be a string with: {possible_eeg_epochs}")

    # Construct dictionary with parameters
    param = {
        'prediction': prediction,
        'dataset_version': dataset_version,
        'eeg_epochs': eeg_epochs,
        'epoch_overlap': epoch_overlap,
        'num_seconds': num_seconds,
        'use_age': use_age,
        'age_scaling': age_scaling,
        'classifier_name': classifier_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'augmentations': augmentations,
        'training_epochs': training_epochs,
        'earlystopping': earlystopping,
        'mc_dropout_enabled': mc_dropout_enabled,
        'mc_dropout_rate': mc_dropout_rate,
        'swa_enabled': swa_enabled,
        'swa_lr': swa_lr,
        'swa_epochs': swa_epochs,
        'swag_enabled': swag_enabled,
        'swag_lr': swag_lr,
        'swag_freq': swag_freq,
    }

    possible_predictions = ('dementia', 'abnormal')

    if param['prediction'] not in possible_predictions:
        raise KeyError(f"Prediction not in in {possible_predictions}, got '{param['prediction']}'")

    return param
