import json
import shutil
from datetime import datetime
from typing import List, Optional
import os
import mlflow

from eegDlUncertainty.data.results.utils_mlflow import get_experiment_name


def check_folder(path, path_ext="figures"):
    full_path = os.path.join(path, path_ext)
    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    return full_path


def get_baseparameters_from_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)

    # Retrieve parameters with type hints
    use_test_set: bool = config.get('test_with_test_set', False)
    result_folder: str = config.get("result_folder")
    prediction: str = config.get('data', {}).get('prediction')
    dataset_version: str = config.get('data', {}).get('version')
    eeg_epochs: str = config.get('data', {}).get('eeg_epochs', 'spread')
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
        'use_test_set': use_test_set,
        'save_path': result_folder,
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


def get_parameters_from_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)

    # Retrieve parameters with type hints
    use_test_set: bool = config.get('test_with_test_set', False)
    result_folder: str = config.get("result_folder")
    prediction: str = config.get('data', {}).get('prediction')
    dataset_version: str = config.get('data', {}).get('version')
    eeg_epochs: str = config.get('data', {}).get('eeg_epochs', 'spread')
    epoch_overlap: int = config.get('data', {}).get('overlapping_epochs')
    num_seconds: int = config.get('data', {}).get('num_seconds')
    use_age: bool = config.get('data', {}).get('use_age')
    age_scaling: str = config.get('data', {}).get('age_scaling')
    classifier_name: str = str(config.get('model', {}).get('name', ''))
    learning_rate: float = float(config.get('hyperparameters', {}).get('learning_rate'))
    batch_size: int = config.get('hyperparameters', {}).get('batch_size')
    augmentations: Optional[List[str]] = config.get('hyperparameters', {}).get('augmentations')
    augment_prob: float = config.get('hyperparameters', {}).get('augment_prob')
    training_epochs: int = config.get('model', {}).get('epochs')
    earlystopping: int = config.get('model', {}).get('earlystopping')

    mc_dropout_enabled: bool = config.get('mc_dropout', {}).get('enabled', False)
    mc_dropout_rate: float = config.get('mc_dropout', {}).get('dropout_rate')
    depth = config.get("model", {}).get("depth")
    cnn_units = config.get("model", {}).get("cnn_units")
    max_kernel_size = config.get("model", {}).get("max_kernel_size")

    swag_start: int = config.get('swag', {}).get('start')
    swag_lr: float = config.get('swag', {}).get('lr')
    swag_freq: int = config.get('swag', {}).get('freq')
    swag_num_models: int = config.get('swag', {}).get('num_models')

    snapshot_epochs: int = config.get('snapshot', {}).get('epochs_per_cycle')
    snapshot_num_cycles: int = config.get('snapshot', {}).get('num_cycles')
    snapshot_lr: float = config.get('snapshot', {}).get('start_lr')
    snapshot_use_best_model: bool = config.get('snapshot', {}).get('use_best_model')

    fge_start_epoch: int = config.get('fge', {}).get('start_epoch')
    fge_num_models: int = config.get('fge', {}).get('num_models')
    fge_epochs_per_cycle: int = config.get('fge', {}).get('epochs_per_cycle')
    fge_cycle_start_lr: float = config.get('fge', {}).get('cycle_start_lr')
    fge_cycle_end_lr: float = config.get('fge', {}).get('cycle_end_lr')

    optuna_experiment: str = config.get('optuna', None)

    possible_eeg_epochs = ['all', 'spread', 'random']

    if eeg_epochs not in possible_eeg_epochs:
        raise KeyError(f"EEG epochs should be a string with: {possible_eeg_epochs}")

    # Construct dictionary with parameters
    param = {
        'use_test_set': use_test_set,
        'save_path': result_folder,
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
        'augment_prob': augment_prob,
        'training_epochs': training_epochs,
        'earlystopping': earlystopping,
        'mc_dropout_enabled': mc_dropout_enabled,
        'mc_dropout_rate': mc_dropout_rate,
        'swag_start': swag_start,
        'swag_lr': swag_lr,
        'swag_freq': swag_freq,
        'swag_num_models': swag_num_models,
        'snapshot_cycle_epochs': snapshot_epochs,
        'snapshot_num_cycles': snapshot_num_cycles,
        'snapshot_lr': snapshot_lr,
        'snapshot_use_best_model': snapshot_use_best_model,
        'fge_start_epoch': fge_start_epoch,
        'fge_num_models': fge_num_models,
        'fge_epochs_per_cycle': fge_epochs_per_cycle,
        'fge_cycle_start_lr': fge_cycle_start_lr,
        'fge_cycle_end_lr': fge_cycle_end_lr,
        'optuna_experiment': optuna_experiment,
        'depth': depth,
        'cnn_units': cnn_units,
        'max_kernel_size': max_kernel_size
    }

    possible_predictions = ('dementia', 'abnormal')

    if param['prediction'] not in possible_predictions:
        raise KeyError(f"Prediction not in in {possible_predictions}, got '{param['prediction']}'")

    return param


def setup_experiment_path(save_path, experiment, config_path):
    """
    Sets up and returns the directory path for saving experiment files.

    This method generates a unique directory path by appending the model name and the current timestamp to the base
    save path specified by `self.save_path`. It then creates this directory (if it doesn't already exist) and copies
    the configuration file specified by `self.config_path` into it. The configuration file is assumed to reside in a
    `config_files` directory located in the same directory as this script.

    Returns
    -------
    str
        The path to the newly created directory where experiment files will be saved.

    Notes
    -----
    - The method uses the `os` module to manipulate paths and create directories, and `shutil` for file operations.
    - It is assumed that `self.save_path`, `self.model_name`, and `self.config_path` are already set and point to
    valid locations or files.
    - The unique directory name is generated using the model name and the current timestamp, which provides a simple
      way to keep experiment outputs organized and avoid overwriting previous results.
    - This method does not handle errors that might occur during directory creation or file copying, so additional
      error handling may be necessary depending on the application's requirements.
    """
    folder_name = f"{experiment}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}"
    paths = os.path.join(save_path, folder_name)
    os.makedirs(paths, exist_ok=True)
    shutil.copy(src=os.path.join(os.path.dirname(__file__), "config_files", config_path),
                dst=os.path.join(paths, os.path.basename(config_path)))
    return paths, folder_name


def prepare_experiment_environment(experiment_name):
    """
    Prepares the environment for running a machine learning experiment.

    This method configures the MLflow tracking URI for experiment logging, enables system metrics logging via an
    environment variable, sets the random seeds for Python's built-in `random` module and NumPy to ensure
    reproducibility, and determines the experiment name based on prediction characteristics.

    The MLflow tracking URI is set to a predefined path, and system metrics logging is enabled to record additional
    system-level information during the experiment. Random seeds are synchronized with the instance's `random_state`
    attribute, affecting random number generation in both Python and NumPy, thereby ensuring that the experiment's
    stochastic elements are reproducible.

    Finally, it calls `get_experiment_name` with parameters that specify the type of prediction, whether the
    prediction is pairwise, and if it is a one-class prediction, to dynamically generate and presumably set
    an experiment name based on these characteristics.

    Notes
    -----
    - The method assumes that `mlflow`, `os`, `random`, and `numpy` modules are imported.
    - The `random_state`, `prediction_type`, `pairwise`, and `one_class` attributes must be set before this method
    is called.
    - It's implied that `get_experiment_name` is a function that sets or uses the experiment name in some capacity,
      though the mechanism for this is not specified within this method.
    """
    mlflow.set_tracking_uri("file:///home/tvetern/PhD/dl_uncertainty/results/mlflow/")
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    get_experiment_name(experiment_name=experiment_name)


def create_run_folder(path, index):
    path = os.path.join(path, index)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_function(experiment_path):
    """
    Attempts to delete the specified folder and its contents.

    Notes
    -----
    - This function uses `shutil.rmtree` to delete the folder and all its contents.
    - Error handling is implemented to catch and log exceptions, preventing the function from raising exceptions if
      the folder cannot be deleted (e.g., if the folder does not exist or an error occurs during deletion).
    """
    try:
        shutil.rmtree(experiment_path)
        print(f"Successfully deleted the folder: {experiment_path} due to OOMemory")
    except Exception as e:
        print(f"Error deleting the folder: {experiment_path}. Exception: {e}")
