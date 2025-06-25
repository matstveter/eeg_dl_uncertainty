import os.path

import numpy as np

from scripts.plots.utils import calculate_confidence_interval, get_clean_folders, get_folders_based_on_name, \
    get_remaining_folders_without_name, read_pkl_to_dict


def calculate_ensemble_metrics(result_path, folder_lists, ensemble_name):
    ensemble_folders = get_folders_based_on_name(folder_lists, ensemble_name)

    results_of_interest = 'average_epochs_merge_softmax'

    extension = "ensemble_test_results.pkl"

    brier_list = []
    auc_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    auc_class_0_list = []
    auc_class_1_list = []
    auc_class_2_list = []

    for ensemble_folder in ensemble_folders:
        ensemble_path = os.path.join(result_path, ensemble_folder, extension)

        if not os.path.exists(ensemble_path):
            print(f"Ensemble path {ensemble_path} does not exist. Skipping.")
            continue
        data_dict = read_pkl_to_dict(ensemble_path)

        data = data_dict[results_of_interest]

        auc = data['performance']['auc']
        acc = data['performance']['accuracy']
        precision = data['performance']['precision']
        recall = data['performance']['recall']
        brier = data['uncertainty']['brier']

        auc_class_0 = data['performance']['auc_class_0']
        auc_class_1 = data['performance']['auc_class_1']
        auc_class_2 = data['performance']['auc_class_2']

        brier_list.append(brier)
        auc_list.append(auc)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_class_0_list.append(auc_class_0)
        auc_class_1_list.append(auc_class_1)
        auc_class_2_list.append(auc_class_2)

    all_metrics = {'brier': np.array(brier_list),
                   'auc': np.array(auc_list),
                   'acc': np.array(acc_list),
                   'precision': np.array(precision_list),
                   'recall': np.array(recall_list),
                   'auc_class_0': np.array(auc_class_0_list),
                   'auc_class_1': np.array(auc_class_1_list),
                   'auc_class_2': np.array(auc_class_2_list)}

    for key, val in all_metrics.items():
        mean, conf_interval = calculate_confidence_interval(metric_list=val)
        if key == "acc":
            mean *= 100
            conf_interval *= 100
        print(f"{key.upper()}: {mean:.2f} +/- {conf_interval:.2f}")


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/3/"
    ensemble_to_calculate = ["WEIGHT", "MCDROPOUT", "AUGMENTATION", "DEPTH", "BAGGING", "FGE", "SNAPSHOT", "SWAG"]
    ensemble_to_calculate = ["SNAPSHOT", "FGE", "SWAG"]
    # ensemble_to_calculate = ["FGE", "SNAPSHOT"]
    ensemble_to_calculate = ['DEEP_AUG']

    print("========================================")
    for ens in ensemble_to_calculate:
        print(f"Ensemble: {ens}")
        print("========================================")
        folder_list = get_clean_folders(res_path=res_path)
        folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")
        calculate_ensemble_metrics(result_path=res_path, folder_lists=folder_list, ensemble_name=ens)
        print("\n")
        print("========================================")
