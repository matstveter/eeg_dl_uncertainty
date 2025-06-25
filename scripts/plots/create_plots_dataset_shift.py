import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd

from scripts.plots.plot_util import bandstop_plot, investigate_age_effect, investigate_class, make_legend_plot, \
    plot_baseline_drift, \
    set_paper_plot_style, \
    static_plot
from scripts.plots.utils import create_a_dataframe, get_clean_folders, get_full_extension, \
    get_remaining_folders_without_name, \
    get_ensemble_names, print_df, read_all_dataset_shifts


def main(result_path, folders):
    full_paths, ensemble_names = get_full_extension(res_path=result_path, folder_list=folders, ret_ensemble_names=True)

    all_df = []
    ensem_names = []
    for i, (path, ens_name) in enumerate(zip(full_paths, ensemble_names)):
        print(f"Processing {ens_name}")
        df = read_all_dataset_shifts(path=path, model_key=i)

        if ens_name in ('AUGMENTATION', 'BAGGING', 'DEPTH', 'SNAPSHOT'):
            # convert to title case
            ens_name = ens_name.title()
        elif ens_name == 'WEIGHT':
            ens_name = 'Deep Ensemble'
        elif ens_name == 'MCDROPOUT':
            ens_name = 'MC Dropout'

        df["ensemble_type"] = ens_name

        if ens_name not in ensem_names:
            # Add the ensemble name to the list
            ensem_names.append(ens_name)

        # if ens_name in ['Augmentation', 'Deep Ensemble']:
        all_df.append(df)

        # if i == 4:
        #     break

    master_df = pd.concat(all_df, ignore_index=True)

    save_path = '/home/tvetern/PhD/dl_uncertainty/figures2/'

    shifts = master_df["shift_name"].unique()
    # Remove shifts that have age in the name
    shifts = [s for s in shifts if "age" not in s]
    # Remove 'baseline'
    shifts = [s for s in shifts if "baseline" not in s]
    # Remove 'bandstop'
    shifts = [s for s in shifts if "bandstop" not in s]
    # We only need one bandstop shift
    shifts.append("bandstop")

    metrics = ["auc", "f1", "accuracy", "brier", "ece", "recall", "precision",
               "auc_class_0", "auc_class_1", "auc_class_2",
               "f1_class_0", "f1_class_1", "f1_class_2",
               "recall_class_0", "recall_class_1", "recall_class_2",
               "precision_class_0", "precision_class_1", "precision_class_2",
               "brier_class_0", "brier_class_1", "brier_class_2"]

    # metrics = ["auc", "f1", "accuracy"]
    metrics = ['auc', 'brier', 'ece']
    # metrics = ['auc_class_0', 'auc_class_1', 'auc_class_2']
    shifts = ['bandstop']
    # metrics = ['auc_class_0', 'auc_class_2']

    palette = sns.color_palette("colorblind", n_colors=len(ensem_names))

    for shift in shifts:
        print("#############################################################")
        print("Processing shift: ", shift)
        print("#############################################################")
        for metric in metrics:
            metric_path = os.path.join(save_path, f"{metric}")
            os.makedirs(metric_path, exist_ok=True)

            print(f"Processing metric: {metric}")
            if "bandstop" in shift:
                bandstop_plot(df=master_df, metric=metric, save_path=metric_path, palette=palette, no_legend=True)
            else:
                static_plot(df=master_df, hue_order=ensem_names, palette=palette,
                            shift_name=shift, metric=metric, save_path=metric_path, no_legend=True)

        print("*************************************************************")
        print("*************************************************************")

    # for shift in shifts:
    #     print("#############################################################")
    #     print("Processing shift: ", shift)
    #     print("#############################################################")
    #     for metric in metrics:
    #         metric_path = os.path.join(save_path, f"{metric}")
    #         os.makedirs(metric_path, exist_ok=True)
    #
    #         print(f"Processing metric: {metric}")
    #         if "bandstop" in shift:
    #             continue
    #         else:
    #             static_plot(df=master_df, hue_order=ensem_names, palette=palette,
    #                         shift_name=shift, metric=metric, save_path=metric_path, no_axis=True, no_legend=True)
    #
    #     print("*************************************************************")
    #     print("*************************************************************")

    # make_legend_plot(ensemble_types=ensem_names, palette=palette, save_path=save_path, ncol=2)
    # make_legend_plot(ensemble_types=ensem_names, palette=palette, save_path=save_path, ncol=3)
    # make_legend_plot(ensemble_types=ensem_names, palette=palette, save_path=save_path, ncol=4)


if __name__ == '__main__':
    # Set the style for the plots
    set_paper_plot_style()
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    folder_list = get_clean_folders(res_path)
    folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")
    folder_list.sort()
    main(result_path=res_path, folders=folder_list)
