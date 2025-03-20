import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd

from scripts.plots.utils import create_a_dataframe, get_clean_folders, get_full_extension, \
    get_remaining_folders_without_name, \
    get_ensemble_names, read_all_dataset_shifts


def plot_shift(df, key):
    shift_df = df[df["shift_name"].str.contains(key, case=False)]
    baseline_df = df[df["shift_name"] == "baseline"]
    combined_df = pd.concat([baseline_df, shift_df], ignore_index=True)

    # sns.boxplot(x="shift_intensity", y="brier", hue="ensemble_type", data=combined_df)
    # plt.title(f"{key}")
    # plt.show()
    #
    sns.boxplot(x="shift_intensity", y="auc", hue="ensemble_type", data=combined_df)
    plt.title(f"{key}")
    plt.show()


def interpolation_plots(df):
    # interp_df = df[df["shift_name"].str.contains("interpolate", case=False)]
    # baseline_df = df[df["shift_name"] == "baseline"]
    # combined_df = pd.concat([baseline_df, interp_df],
    #                         ignore_index=True)
    #
    # sns.boxplot(x="shift_intensity", y="brier", hue="ensemble_type", data=combined_df)
    # plt.show()
    # sns.boxplot(x="shift_intensity", y="auc", hue="ensemble_type", data=combined_df)
    # plt.show()

    plot_shift(df=df, key="interpolate")


def loop_through_shifts(df, shift_key):
    first_sort = df[df["shift_name"].str.contains(shift_key, case=False)]
    baseline_df = df[df["shift_name"] == "baseline"]

    combined_df = pd.concat([baseline_df, first_sort], ignore_index=True)

    plot_shift(df=combined_df, key=shift_key)


def bandstop_plots(df):
    bandstop_df = df[df["shift_name"].str.contains("bandstop", case=False)]
    baseline_df = df[df["shift_name"] == "baseline"]
    bandstop_df = bandstop_df[bandstop_df["shift_intensity"] == "1.0"]

    # Combine baseline and bandstop results into a single DataFrame.
    combined_df = pd.concat([baseline_df, bandstop_df], ignore_index=True)

    # Get the unique shift names in order of appearance
    shift_names = list(combined_df["shift_name"].unique())
    if "baseline" in shift_names:
        shift_names.remove("baseline")
    # Place baseline at the beginning
    shift_order = ["baseline"] + shift_names

    # Create the boxplot.
    sns.boxplot(x="shift_name", y="brier", hue="ensemble_type", data=combined_df, order=shift_order)
    plt.show()

    sns.boxplot(x="shift_name", y="auc", hue="ensemble_type", data=combined_df, order=shift_order)
    plt.show()


def create_classwise_metric_plot(df):
    metrics_long_df = df.melt(
        value_vars=[
            "brier_class_0", "brier_class_1", "brier_class_2",
            "auc_class_0", "auc_class_1", "auc_class_2"
        ],
        var_name="metric_class",
        value_name="value"
    )

    metrics_long_df[['metric', 'class']] = metrics_long_df['metric_class'].str.rsplit("_class_", expand=True)

    # Map class indices to meaningful labels
    class_labels = {'0': 'Normal', '1': 'MCI', '2': 'Dementia'}
    metrics_long_df['class'] = metrics_long_df['class'].map(class_labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    metrics = ['brier', 'auc']
    titles = ['Brier Score', 'AUC']

    for ax, metric, title in zip(axes, metrics, titles):
        sns.boxplot(
            ax=ax,
            x="class",
            y="value",
            data=metrics_long_df[metrics_long_df["metric"] == metric],
            order=["Normal", "MCI", "Dementia"],  # specify order explicitly
            showfliers=False
        )
        ax.set_title(f"{title} per Class")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel(title)

    plt.tight_layout()
    plt.show()


def create_baseline_plot(df):
    baseline_df = df[df["shift_name"] == "baseline"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot AUC
    sns.boxplot(ax=axes[0], x="shift_name", y="auc", hue="ensemble_type", data=baseline_df)
    axes[0].set_title("AUC")
    axes[0].set_xlabel("Ensemble Type")
    axes[0].set_ylabel("AUC")

    # Plot Brier Score
    sns.boxplot(ax=axes[1], x="shift_name", y="brier", hue="ensemble_type", data=baseline_df)
    axes[1].set_title("Brier Score")
    axes[1].set_xlabel("Ensemble Type")
    axes[1].set_ylabel("Brier Score")
    axes[1].get_legend().remove()

    # Plot ECE
    sns.boxplot(ax=axes[2], x="shift_name", y="ece", hue="ensemble_type", data=baseline_df)
    axes[2].set_title("ECE")
    axes[2].set_xlabel("Ensemble Type")
    axes[2].set_ylabel("ECE")
    axes[2].get_legend().remove()

    plt.tight_layout()
    plt.show()


def main(result_path, folders):
    full_paths, ensemble_names = get_full_extension(res_path=result_path, folder_list=folders, ret_ensemble_names=True)

    all_df = []
    i = 0
    for path, ens_name in zip(full_paths, ensemble_names):
        res = read_all_dataset_shifts(path=path)
        df = create_a_dataframe(res_dict=res)
        df["ensemble_type"] = ens_name
        all_df.append(df)
        i += 1

        # if i == 3:
        #     break

    master_df = pd.concat(all_df, ignore_index=True)

    # Create the plots
    # create_baseline_plot(df=master_df)
    # create_classwise_metric_plot(df=master_df)
    # bandstop_plots(df=master_df)
    interpolation_plots(master_df)
    # amplitude_plots(df=master_df)
    loop_through_shifts(df=master_df, shift_key="amplitude")
    loop_through_shifts(df=master_df, shift_key="gaussian")
    loop_through_shifts(df=master_df, shift_key="phase")
    loop_through_shifts(df=master_df, shift_key="circular")
    loop_through_shifts(df=master_df, shift_key="baseline_d")
    loop_through_shifts(df=master_df, shift_key="timewarp")
    loop_through_shifts(df=master_df, shift_key="peak")


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/old_exp/new/"
    folder_list = get_clean_folders(res_path)
    folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")

    main(result_path=res_path, folders=folder_list)
