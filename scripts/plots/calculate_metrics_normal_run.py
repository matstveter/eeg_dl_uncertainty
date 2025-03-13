import os

import numpy as np
import torch
from mypy.memprofile import defaultdict

from scripts.plots.utils import calculate_confidence_interval, get_clean_folders, get_folders_based_on_name, \
    get_metrics, read_pkl_to_dict


def calculate_normal_runs(folders, result_path):
    folders = get_folders_based_on_name(folders, "NORMAL")

    extension = "1/test_set_dict.pkl"
    interesting_keys = ['average', 'majority', 'first_epoch']

    metrics_to_calculate = ['accuracy', 'auc']

    metrics = {}

    for folder in folders:
        abs_path = os.path.join(result_path, folder, extension)
        data_dict = read_pkl_to_dict(path=abs_path)

        for key in interesting_keys:
            if key not in metrics:
                metrics[key] = {}

            for k, v in data_dict[key].items():
                if k in metrics_to_calculate:
                    if k not in metrics[key]:
                        metrics[key][k] = [v]
                    else:
                        metrics[key][k].append(v)

    for key, val in metrics.items():
        print("----------- Key:", key, "-----------")
        for k, v in val.items():
            print("Metric:", k)
            mean, conf_interval = calculate_confidence_interval(metric_list=v)
            print(f"Confidence interval: {mean} +/- {conf_interval}")


def calculate_metrics(subject_keys, predictions, true_classes):

    subject_preds = defaultdict(list)
    subject_labels = {}

    for subject_keys, pred, true_class in zip(subject_keys, predictions, true_classes):
        subject_preds[subject_keys].append(pred)
        if subject_keys not in subject_labels:
            subject_labels[subject_keys] = true_class

    before_softmax = []
    after_softmax = []

    y_one_hot = []

    for key, preds in subject_preds.items():
        # Mean of the predictions
        mean_logits = torch.mean(torch.stack(preds), dim=0)
        before_softmax.append(mean_logits)

        # Softmax of the preds, not the mean_logits
        softm = torch.nn.functional.softmax(torch.stack(preds), dim=1)
        # Then average the softmax into one prediction
        mean_softmax = torch.mean(softm, dim=0)
        after_softmax.append(mean_softmax.numpy())

        y_one_hot.append(subject_labels[key].numpy())

    # Do softmax
    before_softmax = torch.nn.functional.softmax(torch.stack(before_softmax), dim=1)

    # numpy conversion
    before_softmax = np.array(before_softmax)
    after_softmax = np.array(after_softmax)
    y_one_hot = np.array(y_one_hot)

    before_acc, before_auc = get_metrics(y_one_hot=y_one_hot, y_prob=before_softmax)
    after_acc, after_auc = get_metrics(y_one_hot=y_one_hot, y_prob=after_softmax)

    return before_acc, before_auc, after_acc, after_auc


def calculate_metrics_after(folders, result_path):
    folders = get_folders_based_on_name(folders, "NORMAL")

    extension = "1/test_set_dict.pkl"

    accuracy_logits = []
    auc_logits = []
    accuracy_softmax = []
    auc_softmax = []

    for folder in folders:
        abs_path = os.path.join(result_path, folder, extension)
        data_dict = read_pkl_to_dict(path=abs_path)

        subjects = data_dict['subject_keys']
        predictions = data_dict['predictions']['y_pred_tensor']
        y_true = data_dict['predictions']['y_true_tensor']

        log_acc, log_auc, sof_acc, sof_auc = calculate_metrics(subjects, predictions, y_true)

        accuracy_logits.append(log_acc)
        auc_logits.append(log_auc)
        accuracy_softmax.append(sof_acc)
        auc_softmax.append(sof_auc)

    print("----------- AVERAGE -----------")
    print("----------- Key: MERGE LOGITS -----------")
    mean, conf_interval = calculate_confidence_interval(metric_list=accuracy_logits)
    print(f"ACCURACY: Confidence interval: {mean} +/- {conf_interval}")
    mean, conf_interval = calculate_confidence_interval(metric_list=auc_logits)
    print(f"AUC: Confidence interval: {mean} +/- {conf_interval}")

    print("----------- Key: MERGE SOFTMAX -----------")
    mean, conf_interval = calculate_confidence_interval(metric_list=accuracy_softmax)
    print(f"ACCURACY: Confidence interval: {mean} +/- {conf_interval}")
    mean, conf_interval = calculate_confidence_interval(metric_list=auc_softmax)
    print(f"AUC: Confidence interval: {mean} +/- {conf_interval}")


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    folder_list = get_clean_folders(res_path)

    print("\n\nUsing the saved metrics, for average, majority and first_epoch")
    calculate_normal_runs(folders=folder_list, result_path=res_path)

    print("\n\n Calculating metrics both before and after logits, only for average:")
    calculate_metrics_after(folders=folder_list, result_path=res_path)
