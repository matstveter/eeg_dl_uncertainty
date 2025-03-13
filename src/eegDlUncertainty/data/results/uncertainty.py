from typing import Dict, List

import numpy as np
import torch
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from eegDlUncertainty.data.results.ece_variations import get_ace, get_sce, get_tace


def ece(probs, targets, bins=10):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md

    Expected calibration error (ECE) is the primary metric for testing calibration
    (Naeini, Cooper, and Hauskrecht 2015). To calculate ECE, we first divide up the interval between 0 and 1 into bins.
    For instance, we might let the bins be [0, 0.1], [0.1, 0.2], … [0.9, 1]. Then we place examples into these bins
    based on the model’s confidence when making the prediction. Often this means taking the max of the model’s
    post-softmax prediction scores. Finally, we take the weighted sum of the absolute difference between the real
    accuracy and the predicted accuracy. The sum is weighted based on the number of examples in each bin.
    Formally, say we have n examples partitioned up into M bins B1, B2, …, BM. Also, let acc(Bm)
    be the average accuracy of examples in the bin and let conf(Bm) be the average confidence of examples in the bin.

    ECE ranges between 0 and 1, with lower scores being better.
    What is considered a strong ECE varies from dataset to dataset. Reading a few papers,
    we get that ImageNet classifiers usually have ECE which varies from 0.01 to 0.08 and a score of 0.02 or
    lower can be considered strong (Guo et al. 2017; Minderer et al. 2021).

    Returns
    -------

    """
    metric = MulticlassCalibrationError(num_classes=3, n_bins=bins, norm="l1")

    if targets.shape[1] == 3:
        targets = np.argmax(targets, axis=1)

    return metric(torch.from_numpy(probs), torch.from_numpy(targets))


def mce(probs, targets, bins=10):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md

    The Maximum Calibration Error (MCE) is similar to ECE but meant for much more sensitive domains
    (Naeini, Cooper, and Hauskrecht 2015). Like ECE, we partition the interval up into bins. However,
    instead of taking a weighted average of calibration score over bins, we take the maximum calibration
    error over bins. In other words MCE aims to reduce the calibration error of the worst bin, with the
    intuition that this prevents catastrophic failure cases while giving up some efficacy on more mundane cases.


    Like ECE, MCE ranges between 0 and 1, with lower scores being better. MCE is much less common than ECE.
    Quickly eyeballing some results gives us that a model with an MCE of 0.1 can be considered strong (Guo et al. 2017).

    Returns
    -------

    """
    metric = MulticlassCalibrationError(num_classes=3, n_bins=bins, norm="max")
    if targets.shape[1] == 3:
        targets = np.argmax(targets, axis=1)

    return metric(torch.from_numpy(probs), torch.from_numpy(targets))


def nll(probs, targets):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md
    Negative log likelihood (or cross-entropy loss) is commonly used for maximizing predictive accuracy.
    However, NLL is also useful for calibration as well; a classic result in statistics shows that NLL is
    minimized precisely when p(y|x) matches the true probability distribution π(y|x)
    (Hastie, Tibshirani, and Friedman 2009). In other words, NLL is minimized at zero when the
    classifier is perfectly calibrated. In addition, a poor classifier can have unbounded NLL.

    Returns
    -------

    """
    # Make sure that the probs are not 0, to avoid log(0)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(targets * np.log(probs), axis=1))


def brier_score(probs, targets):
    """
    From: https://github.com/centerforaisafety/Intro_to_ML_Safety/blob/master/Interpretable%20Uncertainty/main.md
    Finally, brier score is a common way to measure the accuracy of probability estimates,
    historically used in measuring forecasting accuracy (Brier 1950).
    It is equivalent to measuring the mean squared error of the probability, as follows.

    Brier score is used in many real-world applications, such as assessing weather, sports, or political predictions.
    Brier score is a “strictly proper scoring rule,” meaning that one can uniquely maximize one’s score by predicting
    the true probabilities. Brier score ranges between 0 and 1, with an optimal model having a score of 0.

    Returns
    -------
    """
    assert probs.shape == targets.shape, f"Shapes of probs {probs.shape} and targets {targets.shape} do not match"
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def get_uncertainty_metrics(probs, targets):
    metrics = {'brier': brier_score(probs=probs, targets=targets),
               'nll': nll(probs=probs, targets=targets),
               'ece': ece(probs=probs, targets=targets).numpy().item(),
               'mce': mce(probs=probs, targets=targets).numpy().item(),
               'sce': get_sce(preds=probs, targets=targets),
               'tace_0.05': get_tace(preds=probs, targets=targets, threshold=0.05),
               'ace': get_ace(preds=probs, targets=targets)}

    return metrics


def compute_classwise_brier(mean_probs: np.ndarray, one_hot_target: np.ndarray, targets: np.ndarray):
    """ Compute the Brier score for each class.

    Args:
        mean_probs (np.ndarray): Shape (num_samples, num_classes), mean probabilities for each sample.
        one_hot_target (np.ndarray): Shape (num_samples, num_classes), one-hot encoded true labels.
        targets (np.ndarray): Shape (num_samples,), integer labels.

    Returns:
        dict: Mean Brier Score per class.
    """
    # Compute Brier score for each sample
    brier_per_sample = np.mean((mean_probs - one_hot_target) ** 2, axis=1)

    # Organize Brier score per class
    classwise_brier: Dict[int, List[float]] = {k: [] for k in np.unique(targets)}
    for i in range(len(targets)):
        classwise_brier[targets[i]].append(brier_per_sample[i])

    # Compute mean Brier score per class
    return {k: np.mean(v) if len(v) > 0 else np.nan for k, v in classwise_brier.items()}


def compute_classwise_predictive_entropy(mean_probs: np.ndarray, targets: np.ndarray):
    """Compute Predictive Entropy per class.

    Args:
        mean_probs (np.ndarray): Shape (num_samples, num_classes), mean probabilities for each sample.
        targets (np.ndarray): Shape (num_samples,), integer labels.

    Returns:
        dict: Mean Predictive Entropy per class.
    """
    # Compute entropy per sample
    entropies = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

    # Organize entropy per class
    classwise_entropy: Dict[int, List[float]] = {k: [] for k in np.unique(targets)}
    for i in range(len(targets)):
        classwise_entropy[targets[i]].append(entropies[i])

    # Compute mean entropy per class
    return {k: np.mean(v) if len(v) > 0 else np.nan for k, v in classwise_entropy.items()}


def compute_classwise_variance(mean_probs: np.ndarray, targets: np.ndarray):
    """Compute Predictive Variance per class.

    Args:
        mean_probs (np.ndarray): Shape (num_samples, num_classes), mean probabilities for each sample.
        targets (np.ndarray): Shape (num_samples,), integer labels.

    Returns:
        dict: Mean Predictive Variance per class.
    """
    # Compute variance per class (across samples, not ensembles)
    variance_per_sample = np.var(mean_probs, axis=0)  # Shape: (num_classes,)

    # Organize variance per class
    classwise_variance: Dict[int, List[float]] = {k: [] for k in np.unique(targets)}
    for i in range(len(targets)):
        classwise_variance[targets[i]].append(variance_per_sample[targets[i]])

    # Compute mean variance per class
    return {k: np.mean(v) if len(v) > 0 else np.nan for k, v in classwise_variance.items()}


def compute_classwise_uncertainty(mean_probs, one_hot_target, targets):
    return {"brier": compute_classwise_brier(mean_probs=mean_probs, one_hot_target=one_hot_target, targets=targets),
            "predictive_entropy": compute_classwise_predictive_entropy(mean_probs=mean_probs, targets=targets),
            "variance": compute_classwise_variance(mean_probs=mean_probs, targets=targets)}


def calculate_performance_metrics(y_pred_class, y_true_one_hot, y_true_class, y_pred_prob=None):

    if y_pred_prob is not None:
        try:
            auc = roc_auc_score(y_true=y_true_one_hot, y_score=y_pred_prob, multi_class="ovr", average="weighted")
        except ValueError:
            auc = np.nan
        
        try: 
            auc_per_class = roc_auc_score(
                y_true=y_true_one_hot,
                y_score=y_pred_prob,
                multi_class="ovr",
                average=None
            )
            temp_dict = {}
            for idx, auc_score in enumerate(auc_per_class):
                temp_dict[f"auc_class_{idx}"] = auc_score
        except ValueError:
            temp_dict = {}
            for idx in range(3):
                temp_dict[f"auc_class_{idx}"] = np.nan
    else:
        auc = np.nan
        temp_dict = {'auc_class_0': np.nan, 'auc_class_1': np.nan, 'auc_class_2': np.nan}

    return {'accuracy': accuracy_score(y_true=y_true_class, y_pred=y_pred_class),
            'precision': precision_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted", zero_division=0),
            'recall': recall_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted", zero_division=0),
            'f1': f1_score(y_true=y_true_class, y_pred=y_pred_class, average="weighted", zero_division=0),
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_true=y_true_class, y_pred=y_pred_class),
            **temp_dict
            }
