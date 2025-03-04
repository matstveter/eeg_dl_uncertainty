import numpy as np
import torch
from torchmetrics.classification import MulticlassCalibrationError

import numpy as np


def get_sce(preds, targets, n_bins=10):
    """
    Computes Static Calibration Error (SCE).

    Args:
        preds (np.array): Shape (num_samples, num_classes), predicted probabilities.
        targets (np.array): Shape (num_samples,), integer class labels.
        n_bins (int): Number of bins for calibration.

    Returns:
        float: SCE (Lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    n_objects, n_classes = preds.shape
    res = 0.0

    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(cur_class_conf > bin_lower, cur_class_conf <= bin_upper)

            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = np.mean(targets[in_bin] == cur_class).astype(float)  # Convert boolean to float
                avg_confidence_in_bin = np.mean(cur_class_conf[in_bin])
                delta = np.abs(avg_confidence_in_bin - bin_acc)
                res += delta * bin_size / (n_objects * n_classes)

    return res


def get_tace(preds, targets, n_bins=10, threshold=1e-3):
    """
    Computes Thresholded Adaptive Calibration Error (TACE).

    Args:
        preds (np.array): Shape (num_samples, num_classes), predicted probabilities.
        targets (np.array): Shape (num_samples,), integer class labels.
        n_bins (int): Number of bins.
        threshold (float): Minimum confidence threshold to consider a sample.

    Returns:
        float: TACE (Lower is better)
    """
    n_objects, n_classes = preds.shape
    res = 0.0

    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        # Sort by confidence
        sorted_indices = cur_class_conf.argsort()
        targets_sorted = targets[sorted_indices]
        cur_class_conf_sorted = np.sort(cur_class_conf)

        # Apply thresholding
        valid_indices = cur_class_conf_sorted > threshold
        targets_sorted = targets_sorted[valid_indices]
        cur_class_conf_sorted = cur_class_conf_sorted[valid_indices]

        if len(cur_class_conf_sorted) == 0:
            continue  # Skip if no samples exceed threshold

        bin_size = len(cur_class_conf_sorted) // n_bins

        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            bin_end_ind = bin_start_ind + bin_size if bin_i < n_bins - 1 else len(targets_sorted)
            bin_size = bin_end_ind - bin_start_ind  # Extend last bin

            if bin_size == 0:
                continue  # Skip empty bins

            bin_acc = np.mean(targets_sorted[bin_start_ind: bin_end_ind] == cur_class).astype(float)
            avg_confidence_in_bin = np.mean(cur_class_conf_sorted[bin_start_ind: bin_end_ind])
            delta = np.abs(avg_confidence_in_bin - bin_acc)
            res += delta * bin_size / (n_objects * n_classes)

    return res


def get_ace(preds, targets, n_bins=10):
    """
    Computes Adaptive Calibration Error (ACE).

    Args:
        preds (np.array): Shape (num_samples, num_classes), predicted probabilities.
        targets (np.array): Shape (num_samples,), integer class labels.
        n_bins (int): Number of bins.

    Returns:
        float: ACE (Lower is better)
    """
    return get_tace(preds, targets, n_bins, threshold=0)  # ACE is TACE with threshold=0

