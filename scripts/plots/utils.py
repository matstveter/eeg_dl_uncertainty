import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef,
                             precision_score, recall_score, confusion_matrix, f1_score)


def print_df(df):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(df)


def check_if_folder_exists(path, ood):
    """ Check if the folder exists and has more than 10 files which is the number of dataset shifts"""
    if not os.path.exists(path):
        return False
    else:
        if not ood:
            if len(os.listdir(path)) < 11:
                return False
        else:
            if len(os.listdir(path)) < 4:
                return False
    return True


def get_ensemble_results(res_path, folder_list):
    ensemble_results = []
    for folder in folder_list:
        path = os.path.join(res_path, folder, "ensemble_test_results.pkl")
        if os.path.exists(path):
            ensemble_results.append(path)
    return ensemble_results


def get_full_extension(res_path, folder_list, ood=False, ret_ensemble_names=False):
    """ Get the full path to the datashift extension for the given folders"""
    ext = ""
    if ood:
        ext = "figures/"

    abs_path_list = []
    ensemble_names = []
    for folder in folder_list:
        ensemble_name = folder.split("_")[1]

        if ood:
            path = os.path.join(res_path, folder, ext)
            if check_if_folder_exists(path, ood=ood):
                abs_path_list.append(path)
        else:
            path = os.path.join(res_path, folder)
            abs_path_list.append(path)

        ensemble_names.append(ensemble_name)

    if ret_ensemble_names:
        return abs_path_list, ensemble_names
    else:
        return abs_path_list


def get_ensemble_names(folder_list):
    """ Sort the folders based on the epoch number

    Parameters
    ----------
    folder_list: list
        List of folder names

    Returns
    -------
    The sorted list of folder names

    """
    ensemble_list = []
    for folder in folder_list:
        ens_name = folder.split("_")[1]
        if ens_name not in ensemble_list and ens_name is not None:
            ensemble_list.append(ens_name)

    return ensemble_list


def get_clean_folders(res_path):
    """ Get all folders in the result path that are not 'best', 'data', 'mlflow', 'Old', 'old_optuna'
    This function is tailored to the results folder structure of the dl_uncertainty project, and will not work for
    other folder structures.

    Parameters
    ----------
    res_path: str
        Path to the results folder

    Returns
    -------
    A list of the folder names remaining after removing the non-result folders

    """
    non_result_folders = ['best', 'data', 'mlflow', 'Old', 'old_optuna', '.trash', 'unfinished_exp', '1', '2', '3',
                          'backup', 'old_exp']

    folders = os.listdir(res_path)
    for folder in non_result_folders:
        if folder in folders:
            folders.remove(folder)
    return folders


def get_folders_based_on_name(folder_list, name):
    """ Get the folder based on the name of the folder

    Parameters
    ----------
    folder_list: list
        List of folder names
    name: str
        Name of the folder to get

    Returns
    -------
    The folder name that contains the name

    """

    matching_folders = []

    for folder in folder_list:
        if name in folder:
            matching_folders.append(folder)

    if len(matching_folders) == 0:
        raise ValueError(f"No folder with name {name} found")

    return matching_folders


def get_remaining_folders_without_name(folder_list, name):
    remaining_folders = []

    for folder in folder_list:
        if name not in folder:
            remaining_folders.append(folder)

    return remaining_folders


def read_pkl_to_dict(path):
    with open(path, 'rb') as f:
        return dict(pickle.load(f))


def calculate_confidence_interval(metric_list, alpha=0.05):
    """
    Calculate the confidence interval of a list of metrics, using the t-distribution, and the standard error of the mean

    Parameters
    ----------
    metric_list: list
        List of metrics to calculate the confidence interval for
    alpha: float
        defaults to 0.05, the significance level of the confidence interval

    Returns
    -------
        mean and confidence interval

    """
    if len(metric_list) < 2:
        raise ValueError("Need at least two values to calculate the confidence interval")

    # Check for nan values
    if np.isnan(metric_list).any():
        raise ValueError("The metric list contains nan values")

    mean = np.mean(metric_list)
    sem = st.sem(metric_list)

    t_crit = st.t.ppf(1 - alpha / 2, len(metric_list) - 1)

    return mean, t_crit * sem


def format_ci(lst, alpha=0.05, fmt="${:.2f} \pm {:.2f}$"):
    mean, h = calculate_confidence_interval(lst, alpha=alpha)
    return fmt.format(mean, h)


def get_metrics_(y_one_hot, y_prob):
    auc = roc_auc_score(y_true=y_one_hot, y_score=y_prob, multi_class="ovr", average="weighted")
    accuracy = np.mean(np.argmax(y_prob, axis=1) == np.argmax(y_one_hot, axis=1)) * 100

    # Calculate auc per class
    auc_per_class = roc_auc_score(
        y_true=y_one_hot,
        y_score=y_prob,
        multi_class="ovr",
        average=None
    )

    y_true = np.argmax(y_one_hot, axis=1)
    y_pred = np.argmax(y_prob, axis=1)

    # precision per class
    precision = precision_score(y_true=y_true, y_pred=y_pred, labels=np.unique(y_true), average=None, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None, labels=np.unique(y_true), zero_division=0)

    return accuracy, auc, auc_per_class, precision, recall


def read_all_dataset_shifts(path, model_key):
    """ Read all the dataset shifts from the given path and return a dictionary with the results.

    Each of the dataset shift has the same structure in the end with shift intensity [0.1, 0.25 ...] and the results.
    So this function flattens all the dataset shifts into one dictionary, where the key is the dataset shift and the
    value is the dictionary with shift intensity.

    Parameters
    ----------
    path: str
        path to the directory containing the dataset

    Returns
    -------
        new_dict: dict
            dictionary with the dataset shifts as keys and the results as values

    """
    dataset_paths = ['dataset_shifts', 'dataset_shifts_without_age']

    df_rows = []

    for dataset_path in dataset_paths:

        if dataset_path == "dataset_shifts_without_age":
            ext = "_without_age"
        else:
            ext = ""

        path_ = os.path.join(path, dataset_path)

        if not os.path.exists(path_):
            raise ValueError(f"Path {path_} does not exist")

        dataset_shifts = os.listdir(path_)
        dataset_shifts.sort()

        for shift in dataset_shifts:
            data_dict = read_pkl_to_dict(os.path.join(path_, shift))

            shift_t = shift.split("_")[0]

            if shift_t == "baseline.pkl":
                res_dict = data_dict[0.0]['average_epochs_merge_softmax']
                t_d = get_shift_metrics(res_dict)
                t_d["shift_name"] = f"baseline{ext}"
                t_d["shift_intensity"] = 0.0
                df_rows.append(t_d)
            elif shift_t in ["interpolate.pkl", "channel"]:
                if shift_t == "interpolate.pkl":
                    name = "interpolate"
                    res_dict = data_dict
                else:
                    name = "rotation"
                    res_dict = data_dict['rotate_channels']

                for key, value in res_dict.items():
                    res_dict = value['average_epochs_merge_softmax']
                    t_d = get_shift_metrics(res_dict)
                    t_d["shift_name"] = f"{name}{ext}"
                    t_d["shift_intensity"] = key
                    df_rows.append(t_d)
            elif shift_t == "bandpass":
                for shift_name, res_dict in data_dict.items():
                    for shift_intensity, results in res_dict.items():
                        res_dict = results['average_epochs_merge_softmax']
                        t_d = get_shift_metrics(res_dict)
                        band = shift_name.split("_")[0]
                        t_d["shift_name"] = f"bandstop_{band}{ext}"
                        t_d["shift_intensity"] = shift_intensity
                        df_rows.append(t_d)
            elif shift_t == "baseline":
                for shift_, results in data_dict.items():
                    res_dict = results[1.0]['average_epochs_merge_softmax']

                    shift_name = shift_.split("_")
                    max_drift = shift_name[4]
                    num_sinus = shift_name[-1]

                    if float(max_drift) < 0.1:
                        continue

                    t_d = get_shift_metrics(res_dict)
                    t_d['shift_name'] = f"slow_drift{ext}"
                    # automate the above
                    # t_d['shift_intensity'] = {"max_drift": max_drift, "num_sinus": num_sinus}
                    t_d['shift_intensity'] = f"{max_drift}_{num_sinus}"
                    t_d['max_drift'] = max_drift
                    t_d['num_sinus'] = num_sinus
                    df_rows.append(t_d)

            else:
                for shift_, results in data_dict.items():
                    res_dict = results[1.0]['average_epochs_merge_softmax']
                    t_d = get_shift_metrics(res_dict)
                    name = shift.split("_")
                    name = "_".join(name[:-1])
                    t_d['shift_name'] = f"{name}{ext}"
                    t_d['shift_intensity'] = shift_.split("_")[-1]
                    df_rows.append(t_d)

    df = pd.DataFrame(df_rows)
    df["model_key"] = model_key

    front_cols = ["shift_name", "shift_intensity"]

    # Create a new list of columns: first your front_cols, then everything else
    all_cols = front_cols + [col for col in df.columns if col not in front_cols]

    # Reorder the DataFrame
    return df[all_cols]


def create_a_dataframe(res_dict):
    df_rows = []

    for shift_name, shift_intensities in res_dict.items():
        for sh_i, ensemble_methods in shift_intensities.items():
            for ens_method, results in ensemble_methods.items():

                # We simplify this to only include the average_epochs_merge_softmax
                if ens_method == "average_epochs_merge_softmax":
                    # Extract metrics into a flat dict
                    t_d = get_shift_metrics(results)

                    # Add identifying information so you know which shift/ensemble these metrics belong to
                    t_d["shift_name"] = shift_name
                    t_d["shift_intensity"] = str(sh_i)
                    # t_d["ensemble_method"] = ensemble_remapping.get(ens_method, ens_method)

                    # Append the dictionary as a single row
                    df_rows.append(t_d)
                else:
                    continue
    df = pd.DataFrame(df_rows)

    # Identify the columns you want to place first
    # front_cols = ["shift_name", "shift_intensity", "ensemble_method"]
    front_cols = ["shift_name", "shift_intensity"]

    # Create a new list of columns: first your front_cols, then everything else
    all_cols = front_cols + [col for col in df.columns if col not in front_cols]

    # Reorder the DataFrame
    df = df[all_cols]

    return df


def get_shift_metrics(res_dict):
    temp_dict = {}
    for key, val in res_dict.items():
        if "predictions" not in key:
            for k, v in val.items():
                if "confusion_matrix" not in k:
                    if key == "class_uncertainty":
                        # Here we save it as variance_0 and then the variance for class 0
                        for k2, v2 in v.items():
                            temp_dict[f"{k}_class_{k2}"] = v2
                    else:
                        temp_dict[f"{k}"] = v
        else:
            subject_predicted_probabilities = val['final_subject_probabilities']
            subject_one_hot_labels = val['subject_one_hot_labels']

            try:

                auc_per_class = roc_auc_score(
                    y_true=subject_one_hot_labels,
                    y_score=subject_predicted_probabilities,
                    multi_class="ovr",
                    average=None
                )
                # Present results clearly:
                for idx, auc_score in enumerate(auc_per_class):
                    temp_dict[f"auc_class_{idx}"] = auc_score
            except ValueError as e:
                print(f"Error: {e}")
                print(f"subject_predicted_probabilities: {subject_predicted_probabilities}")
                print(f"subject_one_hot_labels: {subject_one_hot_labels}")

            y_pred = subject_predicted_probabilities.argmax(axis=1)
            y_true = subject_one_hot_labels.argmax(axis=1)

            # compute per-class precision, recall, f1
            precisions = precision_score(
                y_true, y_pred,
                average=None,
                labels=range(subject_one_hot_labels.shape[1]),
                zero_division=0
            )
            recalls = recall_score(
                y_true, y_pred,
                average=None,
                labels=range(subject_one_hot_labels.shape[1]),
                zero_division=0
            )
            f1s = f1_score(
                y_true, y_pred,
                average=None,
                labels=range(subject_one_hot_labels.shape[1]),
                zero_division=0
            )

            # store them
            for idx, (p, r, f) in enumerate(zip(precisions, recalls, f1s)):
                temp_dict[f"precision_class_{idx}"] = p
                temp_dict[f"recall_class_{idx}"] = r
                temp_dict[f"f1_class_{idx}"] = f

    return temp_dict


def create_ood_df(data_dict, ensemble_method="average_epochs_merge_softmax", get_precalculated_metrics=True):
    df_rows = []

    for datasets, dataset_results in data_dict.items():
        if get_precalculated_metrics:
            perf = dataset_results[ensemble_method]['performance']
            unc = dataset_results[ensemble_method]['uncertainty']
            pred = dataset_results[ensemble_method]['predictions']

            row = {
                'accuracy': perf['accuracy'],
                'precision': perf['precision'],
                'recall': perf['recall'],
                'f1': perf['f1'],
                'brier_score': unc['brier'],
                'ece': unc['ece'],
                'nll': unc['nll'],
                'y_prob': pred["final_subject_probabilities"],
                'y_true': pred["subject_class_labels"],
                'max_prob': np.max(pred["final_subject_probabilities"], axis=1),
                'correct': np.argmax(pred["final_subject_probabilities"], axis=1) == np.argmax(
                    pred["subject_one_hot_labels"], axis=1).astype(int),
                'dataset': datasets
            }

            # Calculate accuracy per class
            y_true = pred["subject_class_labels"]
            y_pred = pred['final_subject_class_predictions']

            unique_classes = np.unique(y_true)
            accuracy_per_class = {}

            for cls in unique_classes:
                cls_indices = y_true == cls
                accuracy_cls = accuracy_score(y_true[cls_indices], y_pred[cls_indices])
                accuracy_per_class[cls] = accuracy_cls
                row[f"accuracy_class_{cls}"] = accuracy_cls

            # Calculate MCC
            mcc = matthews_corrcoef(y_true, y_pred)
            row['mcc'] = mcc

            precisions = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
            recalls = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)

            for cls, precision, recall in zip(unique_classes, precisions, recalls):
                row[f"precision_class_{cls}"] = precision
                row[f"recall_class_{cls}"] = recall

            if len(unique_classes) > 2:
                # Calculate per-class AUC
                auc_per_class = roc_auc_score(
                    y_true,
                    pred["final_subject_probabilities"],
                    multi_class="ovr",
                    average=None
                )
                for idx, auc_score in enumerate(auc_per_class):
                    row[f"auc_class_{idx}"] = auc_score

                # Calculate Brier score per class
                # brier_score = np.sum((pred["final_subject_probabilities"] - pred["subject_one_hot_labels"]) ** 2, axis=1)
                # print(brier_score)

            df_rows.append(row)

        else:
            pred = dataset_results[ensemble_method]['predictions']
            y_true_one_hot = pred["subject_one_hot_labels"]
            y_prob = pred["final_subject_probabilities"]
            brier_score = np.sum((y_prob - y_true_one_hot) ** 2, axis=1)
            pred_class = np.argmax(y_prob, axis=1)
            true_class = np.argmax(y_true_one_hot, axis=1)
            max_prob = np.max(y_prob, axis=1)
            correct = np.argmax(y_prob, axis=1) == np.argmax(y_true_one_hot, axis=1)

            for idx in range(len(y_prob)):
                row = {'brier_score': brier_score[idx],
                       'y_prob': y_prob[idx],
                       'y_true_one_hot': y_true_one_hot[idx],
                       'pred_class': pred_class[idx],
                       'true_class': true_class[idx],
                       'max_prob': max_prob[idx],
                       'correct': correct[idx],
                       'dataset': datasets}

                df_rows.append(row)

    return pd.DataFrame(df_rows)


def combine_ood_with_test_results(ood_df, test_dict, ensemble_method="average_epochs_merge_softmax"):
    pred = test_dict[ensemble_method]["predictions"]
    unc = test_dict[ensemble_method]["uncertainty"]
    perf = test_dict[ensemble_method]["performance"]
    brier_per_class = test_dict[ensemble_method]['class_uncertainty']['brier']

    row = {
        'accuracy': perf['accuracy'],
        'precision': perf['precision'],
        'recall': perf['recall'],
        'f1': perf['f1'],
        'brier_score': unc['brier'],
        'ece': unc['ece'],
        'nll': unc['nll'],
        'y_prob': pred["final_subject_probabilities"],
        'y_true': pred["subject_class_labels"],
        'max_prob': np.max(pred["final_subject_probabilities"], axis=1),
        'correct': np.argmax(pred["final_subject_probabilities"], axis=1) == np.argmax(
            pred["subject_one_hot_labels"], axis=1).astype(int),
        'dataset': "test",
    }

    # Calculate accuracy for class 0 and 2
    y_true = pred["subject_class_labels"]
    y_pred = pred['final_subject_class_predictions']

    unique_classes = np.unique(y_true)
    accuracy_per_class = {}

    for cls in unique_classes:
        cls_indices = y_true == cls
        accuracy_cls = accuracy_score(y_true[cls_indices], y_pred[cls_indices])
        accuracy_per_class[cls] = accuracy_cls
        row[f"accuracy_class_{cls}"] = accuracy_cls

    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    row['mcc'] = mcc

    precisions = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)

    for cls, precision, recall in zip(unique_classes, precisions, recalls):
        row[f"precision_class_{cls}"] = precision
        row[f"recall_class_{cls}"] = recall

    # Calculate per-class AUC
    auc_per_class = roc_auc_score(
        y_true,
        pred["final_subject_probabilities"],
        multi_class="ovr",
        average=None
    )
    for idx, auc_score in enumerate(auc_per_class):
        row[f"auc_class_{idx}"] = auc_score

    ood_df = pd.concat([ood_df, pd.DataFrame([row])], ignore_index=True)

    return ood_df


def calculate_ood_metrics(df, ensemble_type,
                          metrics=('brier_score', 'ece', 'recall', 'precision', 'accuracy',
                                   'recall_class_0', 'recall_class_1', 'recall_class_2',
                                   'precision_class_0', 'precision_class_1', 'precision_class_2',
                                   'auc_class_0', 'auc_class_1', 'auc_class_2')):
    datasets = ['test', 'greek', 'mpi', 'tdbrain']
    # datasets = [
    #     'test'
    # ]

    for d in datasets:
        sorted_df = df[df["dataset"] == d]
        sorted_df = sorted_df[sorted_df["ensemble_type"] == ensemble_type]

        print(f"Dataset: {d} Ensemble: {ensemble_type}")
        for metric in metrics:

            if metric not in sorted_df.columns:
                # print(f"Warning: Metric {metric} not found in dataset {d}")
                continue

            values = sorted_df[metric].values

            # Check for NaN values
            if np.isnan(values).any():
                # print(f"Warning: NaN values found in {metric} for dataset {d}")
                continue

            mean, ci = calculate_confidence_interval(values)
            print(f"Metric: {metric:<20} -> Mean: ${mean:.2f}\pm{ci:.2f}$")

        print("-----------------------------------------------------------")


def calculate_softmax_threshold(df, ensemble_type, desired_accuracy=0.70):
    sorted_df = df[df["ensemble_type"] == ensemble_type]
    sorted_df = sorted_df[sorted_df["dataset"] == "test"]

    # Get columns max_prob and correct
    max_prob_runs = sorted_df["max_prob"].values
    correct_runs = sorted_df["correct"].values

    thresholds = []
    coverages = []
    actual_accuracies = []

    for max_prob, correct in zip(max_prob_runs, correct_runs):
        max_prob = np.array(max_prob)
        correct = np.array(correct)

        sorted_indices = np.argsort(-max_prob)
        sorted_probs = max_prob[sorted_indices]
        sorted_correct = correct[sorted_indices]

        cumulative_correct = np.cumsum(sorted_correct)
        total = np.arange(1, len(correct) + 1)
        cumulative_accuracy = cumulative_correct / total

        best_idx = None
        best_coverage = 0

        for idx in range(len(sorted_correct)):
            acc = cumulative_correct[idx] / (idx + 1)
            coverage = (idx + 1) / len(sorted_correct)

            if acc >= desired_accuracy and coverage > best_coverage:
                best_idx = idx
                best_coverage = coverage

        if best_idx is None:
            # If no threshold meets the desired accuracy, append None
            thresholds.append(np.nan)
            coverages.append(0)
            actual_accuracies.append(cumulative_accuracy[-1])
        else:
            threshold = sorted_probs[best_idx]
            coverage = (max_prob >= threshold).mean()
            acc = cumulative_accuracy[best_idx]

            print(f"Desired accuracy: {desired_accuracy} "
                  f"Threshold: {threshold:.2f}, "
                  f"Coverage: {coverage:.2f}, "
                  f"Accuracy: {acc:.2f}",
                  f"Idx: ", best_idx)

            thresholds.append(threshold)
            coverages.append(coverage)
            actual_accuracies.append(acc)

    return {
        "thresholds": thresholds,
        "coverages": coverages,
        "accuracies": actual_accuracies,
        "median_threshold": np.nanmedian(thresholds),
        "mean_threshold": np.nanmean(thresholds),
        "std_threshold": np.nanstd(thresholds),
        "mean_coverage": np.mean(coverages),
        "mean_accuracy": np.mean(actual_accuracies)
    }


def calculate_softmax_threshold_entire_df(df):
    """ Calculate the softmax threshold for the dataframe where the dataset is 'test'. The threshold is calculated
    based on the max probability of the softmax output. The threshold is calculated such that the accuracy of the
    model is at least 0.95, 0.90, and 0.80. The coverage is also calculated for each threshold.

    Parameters
    ----------
    df: pd.DataFrame
        a Dataframe containing the softmax probabilities and the correct predictions

    Returns
    -------
    """

    desired_accuracies = [0.95, 0.90, 0.80]

    for i in range(len(df)):
        row = df.iloc[i]

        if row["dataset"] != "test":
            continue

        max_prob = np.array(row["max_prob"])
        correct = np.array(row["correct"])

        sorted_indices = np.argsort(-max_prob)
        sorted_probs = max_prob[sorted_indices]
        sorted_correct = correct[sorted_indices]

        cumulative_correct = np.cumsum(sorted_correct)
        total = np.arange(1, len(correct) + 1)
        cumulative_accuracy = cumulative_correct / total

        for des_acc in desired_accuracies:
            best_idx = None
            best_coverage = 0

            for j in range(len(sorted_correct)):
                acc = cumulative_correct[j] / (j + 1)
                coverage = (j + 1) / len(sorted_correct)

                if acc >= des_acc and coverage > best_coverage:
                    best_idx = j
                    best_coverage = coverage

                if row['dataset'] == 'BAGGING':
                    print(acc)

            if best_idx is None:
                # If no threshold meets the desired accuracy, append None
                print(row['dataset'], row['ensemble_type'])
                threshold = np.nan
                coverage = 0
                acc = np.nan
            else:
                threshold = sorted_probs[best_idx]
                coverage = (max_prob >= threshold).mean()
                acc = cumulative_accuracy[best_idx]

            df.loc[df.index[i], f"threshold_for_acc_{des_acc}"] = threshold
            df.loc[df.index[i], f"coverage_for_acc_{des_acc}"] = coverage
            df.loc[df.index[i], f"accuracy_for_acc_{des_acc}"] = acc
    return df


def evaluate_thresholds(df):
    # Check if dataframe has any columns with threshold in the name
    threshold_cols = [col for col in df.columns if 'threshold' in col]
    if len(threshold_cols) == 0:
        raise ValueError("Dataframe does not contain any columns with threshold in the name."
                         "Make sure that function: calculate_softmax_threshold_entire_df has been run.")

    desired_accuracies = [float(col.split('_')[-1]) for col in threshold_cols]

    for i in range(len(df)):
        row = df.iloc[i]
        if row["dataset"].lower() == "test":
            # Skip the test dataset
            continue

        # Find matching test row
        matching_test_row = df[
            (df["dataset"].str.lower() == "test") &
            (df["ensemble_type"] == row["ensemble_type"]) &
            (df["model_key"] == row["model_key"])
            ]
        if matching_test_row.empty:
            print(f"Warning: No matching test row found for index {i}")
            continue

        test_row = matching_test_row.iloc[0]  # Suppose one matching row is found

        max_prob = np.array(row["max_prob"])
        correct = np.array(row["correct"]).astype(int)

        for des_acc in desired_accuracies:
            threshold = test_row[f"threshold_for_acc_{des_acc}"]
            coverage = test_row[f"coverage_for_acc_{des_acc}"]

            if coverage == 0.0:
                df.loc[df.index[i], f"ood_accuracy_at_acc_{des_acc}"] = np.nan
                df.loc[df.index[i], f"ood_coverage_at_acc_{des_acc}"] = 0
                continue

            mask = max_prob >= threshold

            if np.sum(mask) == 0:
                acc = np.nan
                coverage = 0
            else:
                acc = correct[mask].mean()
                coverage = mask.mean()

            df.loc[df.index[i], f"ood_accuracy_at_acc_{des_acc}"] = acc
            df.loc[df.index[i], f"ood_coverage_at_acc_{des_acc}"] = coverage

    return df
