import numpy as np
import torch
from torchmetrics.classification import MulticlassCalibrationError


def get_ece(preds, targets, n_bins=15, **args):
    metric = MulticlassCalibrationError(num_classes=3, n_bins=n_bins, norm="l1")
    if targets.shape[1] == 3:
        targets = np.argmax(targets, axis=1)
    return metric(torch.from_numpy(preds), torch.from_numpy(targets)).item()


def get_sce(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    n_objects, n_classes = preds.shape
    res = 0.0
    for cur_class in range(n_classes):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            cur_class_conf = preds[:, cur_class]
            in_bin = np.logical_and(cur_class_conf > bin_lower, cur_class_conf <= bin_upper)

            # cur_class_acc is ground truth probability of chosen class being the correct one inside the bin.
            # NOT fraction of correct predictions in the bin
            # because it is compared with predicted probability
            bin_acc = (targets[in_bin] == cur_class)

            bin_conf = cur_class_conf[in_bin]

            bin_size = np.sum(in_bin)

            if bin_size > 0:
                avg_confidence_in_bin = np.mean(bin_conf)
                avg_accuracy_in_bin = np.mean(bin_acc)
                delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
                res += delta * bin_size / (n_objects * n_classes)
    return res


def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape

    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = np.sort(cur_class_conf)

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins

        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind: bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind: bin_end_ind]
            avg_confidence_in_bin = np.mean(bin_conf)
            avg_accuracy_in_bin = np.mean(bin_acc)
            delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            res += delta * bin_size / (n_objects * n_classes)

    return res


def get_ace(preds, targets, n_bins=15, **args):
    return get_tace(preds, targets, n_bins, threshold=0)
