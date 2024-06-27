import numpy as np
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import OODDataGenerator
from eegDlUncertainty.data.dataset.OODDataset import GreekEEGDataset, MPILemonDataset, TDBrainDataset
from eegDlUncertainty.data.results.dataset_shifts import activation_function
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle


def get_loaders(dataset, device, batch_size):
    """
    This function creates a DataLoader object for a given dataset.

    Parameters:
    dataset (Dataset): The dataset for which the DataLoader is to be created.
    device (Device): The device on which the data loading operations are to be performed.
    batch_size (int): The number of samples per batch to load.

    Returns:
    DataLoader: A DataLoader object for the given dataset.
    """
    greek_data_gen = OODDataGenerator(dataset=dataset, use_age=True, device=device)
    return DataLoader(greek_data_gen, batch_size=batch_size, shuffle=False)


def get_dataset(dataset_version: int, num_seconds: int, age_scaling: str, device, batch_size: int):
    """
    This function creates DataLoader objects for three different datasets: GreekEEGDataset, MPILemonDataset, and
    TDBrainDataset.

    Parameters:
    dataset_version (int): The version of the dataset to be loaded.
    num_seconds (int): The number of seconds of EEG data to be loaded.
    age_scaling (str): The scaling method to be applied to the age data in the dataset.
    device (Device): The device on which the data loading operations are to be performed.
    batch_size (int): The number of samples per batch to load.

    Returns:
    tuple: A tuple containing DataLoader objects for the GreekEEGDataset, MPILemonDataset, and TDBrainDataset.
    """
    greek = GreekEEGDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling)
    mpi = MPILemonDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling)
    tdbrain = TDBrainDataset(dataset_version=dataset_version, num_seconds_eeg=num_seconds, age_scaling=age_scaling)

    return (get_loaders(dataset=greek, device=device, batch_size=batch_size),
            get_loaders(dataset=mpi, device=device, batch_size=batch_size),
            get_loaders(dataset=tdbrain, device=device, batch_size=batch_size))


def single_dataset_experiment(model, data_loader, device):
    """
    This function performs an experiment on a single dataset.

    Parameters:
    model (Model or list of Model): The model or ensemble of models to be used for the experiment.
    data_loader (DataLoader): The DataLoader object for the dataset to be used in the experiment.
    device (Device): The device on which the experiment is to be performed.

    Returns:
    dict: A dictionary containing the performance metrics, uncertainty metrics, and class-wise uncertainty metrics of
     the experiment.
    dict: A dictionary containing the mean logits, all probabilities, probabilities, predictions, and target classes of
    the experiment.
    """
    if not isinstance(model, list):
        print("Testing with the combined EEG")
        logits, targets = model.get_mc_predictions(test_loader=data_loader, device=device, history=None,
                                                   num_forward=50)
    else:
        print("Testing with the ensemble")
        logits = []
        for m in model:
            ensemble_logits, targets = m.get_predictions(loader=data_loader, device=device)
            logits.append(ensemble_logits)
        logits = np.array(logits)  # shape: (num_models, num_samples, num_classes)

    mean_logits = torch.from_numpy(np.mean(logits, axis=0))
    all_predictions = activation_function(logits=torch.from_numpy(logits), ensemble=True)
    probs = activation_function(logits=mean_logits, ensemble=False)
    predictions = activation_function(logits=mean_logits, ensemble=False, ret_prob=False)
    target_classes = np.argmax(targets, axis=1)

    ood_predictions = {"mean_logits": mean_logits,
                       "all_probs": all_predictions,
                       "probs": probs,
                       "predictions": predictions,
                       "target_classes": target_classes}

    performance = calculate_performance_metrics(y_pred_prob=probs, y_pred_class=predictions,
                                                y_true_one_hot=targets, y_true_class=target_classes)

    uncertainty = get_uncertainty_metrics(probs=probs, targets=targets)

    class_uncertainty = compute_classwise_uncertainty(all_probs=all_predictions, mean_probs=probs,
                                                      one_hot_target=targets, targets=target_classes)

    ood_results = {"performance": performance,
                   "uncertainty": uncertainty,
                   "class_uncertainty": class_uncertainty}

    return ood_results, ood_predictions


def ood_experiment(classifiers, dataset_version: int, num_seconds: int, age_scaling: str, device, batch_size: int,
                   save_path: str):
    greek_loader, mpi_loader, tdbrain_loader = get_dataset(dataset_version=dataset_version,
                                                           num_seconds=num_seconds,
                                                           device=device, batch_size=batch_size,
                                                           age_scaling=age_scaling)

    greek_res, greek_pred = single_dataset_experiment(model=classifiers, data_loader=greek_loader, device=device)
    print("Greek results: ", greek_res)
    mpi_res, mpi_pred = single_dataset_experiment(model=classifiers, data_loader=mpi_loader, device=device)
    print("MPI results: ", mpi_res)
    tdbrain_res, tdbrain_pred = single_dataset_experiment(model=classifiers, data_loader=tdbrain_loader, device=device)
    print("TDBrain results: ", tdbrain_res)

    combined_results = {'greek': {'results': greek_res, 'predictions': greek_pred},
                        'mpi': {'results': mpi_res, 'predictions': mpi_pred},
                        'tdbrain': {'results': tdbrain_res, 'predictions': tdbrain_pred}}

    # todo What to plot?

    save_dict_to_pickle(data_dict=combined_results, path=save_path, name="ood_results")
    return combined_results
