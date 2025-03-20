import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_auc_score


def check_if_folder_exists(path, ood):
    """ Check if the folder exists and has more than 10 files which is the number of dataset shifts"""
    if not os.path.exists(path):
        return False
    else:
        if not ood:
            if len(os.listdir(path)) < 11:
                return False
        else:
            if len(os.listdir(path)) < 8:
                return False
    return True


def get_full_extension(res_path, folder_list, ood=False, ret_ensemble_names=False):
    """ Get the full path to the datashift extension for the given folders"""
    if ood:
        ext = "figures/"
    else:
        ext = "dataset_shifts/"

    special_cases = ["FGE", "SNAPSHOT"]
    ext_special = ["ensemble_5", "ensemble_20"]

    abs_path_list = []
    ensemble_names = []
    for folder in folder_list:
        ensemble_name = folder.split("_")[1]
        if folder in special_cases:
            for ext_s in ext_special:
                path = os.path.join(res_path, folder, ext_s, ext)

                if check_if_folder_exists(path, ood=ood):
                    abs_path_list.append(path)
                    ensemble_name = f"{ensemble_name}_{ext_s}"
        else:
            path = os.path.join(res_path, folder, ext)
            if check_if_folder_exists(path, ood=ood):
                abs_path_list.append(path)

        # # There should not be any duplicates, so if bagging exists, the next should be bagging_2, it can be many
        # if ensemble_name in ensemble_names:
        #     i = 1
        #     while True:
        #         temp_name = f"{ensemble_name}_{i}"
        #         if temp_name not in ensemble_names:
        #             break
        #         i += 1
        #     ensemble_name = temp_name

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
    non_result_folders = ['best', 'data', 'mlflow', 'Old', 'old_optuna', '.trash', 'unfinished_exp']

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


def get_metrics(y_one_hot, y_prob):
    auc = roc_auc_score(y_true=y_one_hot, y_score=y_prob, multi_class="ovr", average="weighted")
    accuracy = np.mean(np.argmax(y_prob, axis=1) == np.argmax(y_one_hot, axis=1)) * 100

    return accuracy, auc


def extract_intensity_results(data_dict):
    for k, v in data_dict.items():
        print(k)


def read_all_dataset_shifts(path):
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

    dataset_shifts = os.listdir(path)
    dataset_shifts.sort()

    new_dict = {}

    for shift in dataset_shifts:
        data_dict = read_pkl_to_dict(os.path.join(path, shift))

        if "baseline" in shift and "drift" not in shift:
            new_dict["baseline"] = data_dict
        elif "channel" in shift:
            new_dict['channel_rotation'] = data_dict['rotate_channels']
        elif "interpolate" in shift:
            new_dict['interpolate'] = data_dict
        else:
            for key, value in data_dict.items():
                if "bandpass" in shift:
                    # Change name to bandstop as it is more correct, and then extract the band,
                    band = key.split("_")[0]
                    new_dict[f"bandstop_{band}"] = value
                else:
                    new_dict[key] = value

    return new_dict


def create_a_dataframe(res_dict):
    df_rows = []

    for shift_name, shift_intensities in res_dict.items():
        for sh_i, ensemble_methods in shift_intensities.items():
            for ens_method, results in ensemble_methods.items():

                # We simplify this to only include the average_epochs_merge_softmax
                if ens_method == "average_epochs_merge_logits":
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

            auc_per_class = roc_auc_score(
                y_true=subject_one_hot_labels,
                y_score=subject_predicted_probabilities,
                multi_class="ovr",
                average=None
            )
            # Present results clearly:
            for idx, auc_score in enumerate(auc_per_class):
                temp_dict[f"auc_class_{idx}"] = auc_score

    return temp_dict
