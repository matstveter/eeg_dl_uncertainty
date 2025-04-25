import os

import numpy as np
import torch
from mypy.memprofile import defaultdict
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from scripts.plots.utils import calculate_confidence_interval, get_clean_folders, get_folders_based_on_name, \
    get_metrics_, read_pkl_to_dict


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

    after_softmax = []

    y_one_hot = []

    for key, preds in subject_preds.items():

        # Softmax of the preds, not the mean_logits
        softm = torch.nn.functional.softmax(torch.stack(preds), dim=1)
        # Then average the softmax into one prediction
        mean_softmax = torch.mean(softm, dim=0)
        after_softmax.append(mean_softmax.numpy())

        y_one_hot.append(subject_labels[key].numpy())

    # numpy conversion
    after_softmax = np.array(after_softmax)
    y_one_hot = np.array(y_one_hot)

    return get_metrics_(y_one_hot=y_one_hot, y_prob=after_softmax)


def calculate_metrics_after(folders, result_path):
    folders = get_folders_based_on_name(folders, "NORMAL")

    extension = "1/test_set_dict.pkl"

    accuracy_softmax = []
    auc_softmax = []
    auc_class_0 = []
    auc_class_1 = []
    auc_class_2 = []
    precision_class_0 = []
    precision_class_1 = []
    precision_class_2 = []
    recall_class_0 = []
    recall_class_1 = []
    recall_class_2 = []

    for folder in folders:
        abs_path = os.path.join(result_path, folder, extension)
        data_dict = read_pkl_to_dict(path=abs_path)

        subjects = data_dict['subject_keys']
        predictions = data_dict['predictions']['y_pred_tensor']
        y_true = data_dict['predictions']['y_true_tensor']

        accuracy, auc, auc_class, precision_class, recall_class = calculate_metrics(subjects, predictions, y_true)

        accuracy_softmax.append(accuracy)
        auc_softmax.append(auc)
        auc_class_0.append(auc_class[0])
        auc_class_1.append(auc_class[1])
        auc_class_2.append(auc_class[2])
        precision_class_0.append(precision_class[0])
        precision_class_1.append(precision_class[1])
        precision_class_2.append(precision_class[2])
        recall_class_0.append(recall_class[0])
        recall_class_1.append(recall_class[1])
        recall_class_2.append(recall_class[2])

    # Create a dictionary to store the metrics
    metrics_dict = {
        "accuracy": accuracy_softmax,
        "auc": auc_softmax,
        "auc_class_0": auc_class_0,
        "auc_class_1": auc_class_1,
        "auc_class_2": auc_class_2,
        "precision_class_0": precision_class_0,
        "precision_class_1": precision_class_1,
        "precision_class_2": precision_class_2,
        "recall_class_0": recall_class_0,
        "recall_class_1": recall_class_1,
        "recall_class_2": recall_class_2
    }

    for key, val in metrics_dict.items():
        mean, conf_interval = calculate_confidence_interval(metric_list=val)
        print(f"{key:<20}: ${mean:.2f}\pm{conf_interval:.2f}$")


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    folder_list = get_clean_folders(res_path)

    # print("\n\nUsing the saved metrics, for average, majority and first_epoch")
    # calculate_normal_runs(folders=folder_list, result_path=res_path)

    print("\n\n Calculating metrics both before and after logits, only for average:")
    calculate_metrics_after(folders=folder_list, result_path=res_path)
