import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mlflow

from eegDlUncertainty.experiments.utils_exp import check_folder


def calculate_metrics(all_pred: np.ndarray, all_targets: np.ndarray, save_path) -> dict:
    save_path = check_folder(save_path, path_ext="figures")
    # Variance
    pred_variance = np.var(all_pred, axis=0)

    # Performance accuracy
    mean_predictions = np.mean(all_pred, axis=0)
    targets = np.mean(all_targets, axis=0)
    accuracy = accuracy_score(np.argmax(targets, axis=1), np.argmax(mean_predictions, axis=1))

    # Predictive entropy
    entropy_values = calculate_predictive_entropy(predictions=all_pred)

    # Mutual information

    metrics = {
        'variance': pred_variance,
        'accuracy': accuracy,
        'predictive_entropy': entropy_values
    }
    plot_predictive_entropy(pred_entropy=entropy_values, predictions=all_pred, file_path=save_path)
    return metrics


def plot_predictive_entropy(pred_entropy, predictions, file_path):
    plt.figure(figsize=(20, 12), dpi=300)
    plt.hist(pred_entropy, bins=30, alpha=0.75, color='blue')
    plt.title(f'Histogram of Predictive Entropy: Max = {np.log(predictions.shape[2])}', fontsize=20)
    plt.xlabel('Predictive Entropy', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    full_path = f"{file_path}/histogram_pred_entropy.pdf"
    plt.savefig(full_path, format="pdf")
    mlflow.log_artifact(full_path)


def calculate_predictive_entropy(predictions):
    """
    Calculate the predictive entropy of predictions from multiple stochastic forward passes.

    Parameters
    ----------
    predictions : numpy.ndarray
        An array of shape (num_passes, num_samples, num_classes) containing the model predictions
        from multiple stochastic forward passes.

    Returns
    -------
    numpy.ndarray
        An array containing the predictive entropy for each sample.
    """
    # Calculate mean probability across passes for each class
    mean_probs = np.mean(predictions, axis=0)

    # Compute the entropy for each sample
    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)  # small constant for numerical stability
    return pred_entropy


def mapping_avg_state_dict(averaged_model_state_dict):
    mapped_state_dict = {}
    for key, value in averaged_model_state_dict.items():
        if 'n_averaged' not in key:
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            mapped_state_dict[new_key] = value

    return mapped_state_dict
