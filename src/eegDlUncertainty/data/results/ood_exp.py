import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import OODDataGenerator
from eegDlUncertainty.data.dataset.OODDataset import GreekEEGDataset, MPILemonDataset, TDBrainDataset
from eegDlUncertainty.data.results.dataset_shifts import activation_function
from eegDlUncertainty.data.results.uncertainty import calculate_performance_metrics, compute_classwise_uncertainty, \
    get_uncertainty_metrics
from eegDlUncertainty.data.utils import save_dict_to_pickle
from eegDlUncertainty.experiments.utils_exp import check_folder

FIG_SIZE = (20, 12)
TITLE_FONT = 30
LABEL_FONT = 28
TICK_FONT = 20
# Mapping from class indices to labels
class_labels = {0: "Normal", 1: "MCI", 2: "Dementia"}


def calculate_max_brier_score(num_classes):
    max_prob = np.ones(num_classes) / num_classes
    max_brier = np.sum((max_prob - np.eye(num_classes)) ** 2)
    return max_brier


def create_scatter_plot(probs_pred, target_classes, dataset_name, save_path, jitter=0.15):
    """
    Creates and saves a scatter plot visualizing the Brier scores for predicted probabilities
    compared to the target classes, distinguishing between correct and incorrect predictions.

    Parameters
    ----------
    probs_pred : np.ndarray
        Array of predicted probabilities with shape (n_samples, n_classes).
    target_classes : np.ndarray
        One-hot encoded array of target classes with shape (n_samples, n_classes).
    dataset_name : str
        Name of the dataset to use in the plot title and saved filename.
    save_path : str
        Directory path where the plot will be saved.
    jitter : float, optional
        Amount of random jitter to add to the class labels for better visualization. Default is 0.15.

    Returns
    -------
    None
        The function saves the scatter plot as an EPS file in the specified directory and displays it.

    Notes
    -----
    The function calculates the Brier score for each sample, which is a measure of the accuracy of
    probabilistic predictions. The plot shows the Brier scores for each class, color-coded based on
    whether the predictions are correct or incorrect.
    """
    # Calculate the Brier score for each sample
    brier = np.mean((probs_pred - target_classes) ** 2, axis=1)

    # Get the predicted and true classes
    predictions = np.argmax(probs_pred, axis=1)
    true_classes = np.argmax(target_classes, axis=1)

    # Identify correct and incorrect predictions
    correct_predictions = predictions == true_classes
    # Create a DataFrame for Seaborn
    data = {
        'Class': true_classes + np.random.uniform(-jitter, jitter, size=true_classes.shape),
        'Brier Score': brier,
        'Correct': np.where(correct_predictions, 'Correct', 'Wrong')
    }
    df = pd.DataFrame(data)

    sns.set(style="darkgrid")

    plt.figure(figsize=FIG_SIZE)

    # Use Seaborn scatterplot
    sns.scatterplot(x='Class', y='Brier Score', hue='Correct', data=df,
                    palette={'Correct': 'green', 'Wrong': 'red'}, legend='full', s=125)
    # Draw a horizontal line for the maximum Brier score
    max_brier = calculate_max_brier_score(num_classes=probs_pred.shape[1])
    plt.axhline(y=max_brier, color='blue', linestyle='--', label=f'Max Brier Score')

    plt.xlabel("Class", fontsize=LABEL_FONT)

    # Dynamically set x-ticks based on present classes
    present_classes = np.unique(true_classes)
    present_class_labels = [class_labels[cls] for cls in present_classes]
    plt.xticks(ticks=present_classes, labels=present_class_labels, fontsize=TICK_FONT)

    plt.ylabel("Brier Score", fontsize=LABEL_FONT)
    plt.title(f'{dataset_name}: Performance vs. Uncertainty', fontsize=TITLE_FONT, weight='bold', color='navy')
    plt.legend(title='Prediction', fontsize=TICK_FONT, title_fontsize=LABEL_FONT)
    plt.tight_layout()

    plt.savefig(f"{save_path}/{dataset_name}_OOD.eps", dpi=300, format="eps")
    plt.show()


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


def single_dataset_experiment(model, data_loader, device, dataset_name, save_path):
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

    targets = None
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
    create_scatter_plot(probs_pred=probs, target_classes=targets, dataset_name=dataset_name, save_path=save_path)

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

    save_path = check_folder(path=save_path, path_ext="figures")

    greek_res, greek_pred = single_dataset_experiment(model=classifiers, data_loader=greek_loader, device=device,
                                                      dataset_name="Greek", save_path=save_path)
    print("Greek results: ", greek_res)
    mpi_res, mpi_pred = single_dataset_experiment(model=classifiers, data_loader=mpi_loader, device=device,
                                                  dataset_name="MPI", save_path=save_path)
    print("MPI results: ", mpi_res)
    tdbrain_res, tdbrain_pred = single_dataset_experiment(model=classifiers, data_loader=tdbrain_loader, device=device,
                                                          dataset_name="TDBrain", save_path=save_path)
    print("TDBrain results: ", tdbrain_res)

    combined_results = {'greek': {'results': greek_res, 'predictions': greek_pred},
                        'mpi': {'results': mpi_res, 'predictions': mpi_pred},
                        'tdbrain': {'results': tdbrain_res, 'predictions': tdbrain_pred}}

    # todo What to plot?

    save_dict_to_pickle(data_dict=combined_results, path=save_path, name="ood_results")
    return combined_results


# if __name__ == "__main__":
#     # Example usage
#     num_samples = 10
#     num_classes = 3
#     # true_labels = [0] * num_samples  # Only the 0th class present
#     true_labels = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
#
#     # Generate random probabilities
#     np.random.seed(0)
#     probs = np.random.rand(num_samples, num_classes)
#     probs = probs / np.sum(probs, axis=1, keepdims=True)  # Normalize to sum to 1
#
#     # Generate random target classes
#     # true_labels = np.random.choice(num_classes, num_samples)
#     target_classes = np.eye(num_classes)[true_labels]  # One-hot encoding
#
#     create_scatter_plot(probs, target_classes, dataset_name="Test", save_path=".")
