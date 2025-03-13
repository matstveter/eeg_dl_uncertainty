import os.path

from scripts.plots.utils import calculate_confidence_interval, get_clean_folders, get_folders_based_on_name, \
    get_remaining_folders_without_name, read_pkl_to_dict


def calculate_ensemble_metrics(result_path, folder_lists, ensemble_name):
    ensemble_folders = get_folders_based_on_name(folder_lists, ensemble_name)

    results_of_interest = 'average'

    extension = "ensemble_test_results.pkl"
    special_cases = ['FGE', 'SNAPSHOT']

    if ensemble_name in special_cases:
        ext_1 = ["ensemble_5", "ensemble_20"]

        ensemble_metrics_auc = {}
        ensemble_metrics_acc = {}

        ensemble_brier = {}

        for ensemble_folder in ensemble_folders:
            for ext in ext_1:
                ensemble_path = os.path.join(result_path, ensemble_folder, ext, extension)
                data_dict = read_pkl_to_dict(ensemble_path)

                for key, val in data_dict.items():
                    if results_of_interest in key:
                        if key not in ensemble_metrics_auc:
                            ensemble_metrics_auc[key] = []
                            ensemble_metrics_acc[key] = []
                            ensemble_brier[key] = []

                        ensemble_metrics_auc[key].append(val['performance']['auc'])
                        ensemble_metrics_acc[key].append(val['performance']['accuracy'])
                        ensemble_brier[key].append(val['uncertainty']['brier'])
    else:
        ensemble_metrics_auc = {}
        ensemble_metrics_acc = {}
        
        ensemble_brier = {}

        for ensemble_folder in ensemble_folders:
            ensemble_path = os.path.join(result_path, ensemble_folder, extension)
            data_dict = read_pkl_to_dict(ensemble_path)

            for key, val in data_dict.items():

                if results_of_interest in key:
                    if key not in ensemble_metrics_auc:
                        ensemble_metrics_auc[key] = []
                        ensemble_metrics_acc[key] = []
                        ensemble_brier[key] = []

                    ensemble_metrics_auc[key].append(val['performance']['auc'])
                    ensemble_metrics_acc[key].append(val['performance']['accuracy'])
                    
                    ensemble_brier[key].append(val['uncertainty']['brier'])

    print("\n ----------------- AUC ----------------- ")
    for key, val in ensemble_metrics_auc.items():
        print(f"Key: {key}")
        mean, conf_interval = calculate_confidence_interval(metric_list=val)
        print(f"Confidence interval: {mean} +/- {conf_interval}")

    print("\n ----------------- Accuracy ----------------- ")
    for key, val in ensemble_metrics_acc.items():
        print(f"Key: {key}")
        mean, conf_interval = calculate_confidence_interval(metric_list=val)
        print(f"Confidence interval: {mean} +/- {conf_interval}")
    
    print("\n ----------------- Brier ----------------- ")
    for key, val in ensemble_brier.items():
        print(f"Key: {key}")
        mean, conf_interval = calculate_confidence_interval(metric_list=val)
        print(f"Confidence interval: {mean} +/- {conf_interval}")


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    ensemble_to_calculate = "MCD"

    folder_list = get_clean_folders(res_path)
    folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")
    calculate_ensemble_metrics(result_path=res_path, folder_lists=folder_list, ensemble_name=ensemble_to_calculate)
