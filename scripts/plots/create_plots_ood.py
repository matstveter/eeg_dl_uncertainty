import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import seaborn as sns

from scripts.plots.utils import calculate_ood_metrics, calculate_softmax_threshold, \
    calculate_softmax_threshold_entire_df, combine_ood_with_test_results, \
    create_ood_df, evaluate_thresholds, get_clean_folders, \
    get_ensemble_results, \
    get_full_extension, \
    get_remaining_folders_without_name, print_df, read_pkl_to_dict


def plot_test_results(df, ensemble_type='WEIGHT', boxplot=False):
    df_sort = df[df["ensemble_type"] == ensemble_type]

    if boxplot:
        sns.boxplot(x="true_class", y="brier_score", data=df_sort)
        plt.show()
        return

    # Create numeric labels for classes to allow jittering
    classes = df_sort['true_class'].unique()
    class_to_num = {cls: num for num, cls in enumerate(classes)}

    # Map true_class to numeric for jitter
    df_sort['class_num'] = df_sort['true_class'].map(class_to_num)

    # Add jitter
    jitter_strength = 0.1
    df_sort['class_num_jittered'] = df_sort['class_num'] + np.random.uniform(
        low=-jitter_strength, high=jitter_strength, size=len(df_sort))

    # Define color mapping
    palette = {True: 'green', False: 'red'}

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='class_num_jittered',
        y='brier_score',
        hue='correct',
        palette=palette,
        data=df_sort,
        alpha=0.7
    )

    plt.xticks(ticks=range(len(classes)), labels=['Normal', 'Dementia'])
    plt.xlabel('True Class')
    plt.ylabel('Brier Score')
    plt.title('Brier Score vs. True Class with Jitter')
    plt.legend(title='Correct Prediction', loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_ood_results(df, ensemble_type='WEIGHT', boxplot=False):
    df_sort = df[df["ensemble_type"] == ensemble_type]


def main(result_path, folders):
    # Get the full path to the results and the ensemble names.
    full_paths, ensemble_names = get_full_extension(res_path=result_path, folder_list=folders,
                                                    ood=True, ret_ensemble_names=True)
    test_paths = get_ensemble_results(res_path=result_path, folder_list=folders)

    ood_df_all = []

    # Loop through the full paths and ensemble names. Create df for each ood performance and test set performance.
    for key, (path, test_set_path, ens) in enumerate(zip(full_paths, test_paths, ensemble_names)):
        ood_df = create_ood_df(read_pkl_to_dict(os.path.join(path, "ood_results.pkl")))
        comb_df = combine_ood_with_test_results(ood_df=ood_df,
                                                test_dict=read_pkl_to_dict(os.path.join(path, test_set_path)))
        comb_df["ensemble_type"] = ens
        comb_df["model_key"] = key
        ood_df_all.append(comb_df)

    master_ood_df = pd.concat(ood_df_all, ignore_index=True)
    ensembles = master_ood_df['ensemble_type'].unique()

    # metrics = ['auc']
    # ensembles = ['SWAG']
    metrics = []


    for ens in ensembles:
        if metrics:
            calculate_ood_metrics(df=master_ood_df, metrics=metrics, ensemble_type=ens)
        else:
            calculate_ood_metrics(df=master_ood_df, ensemble_type=ens)

    # master_ood_df = calculate_softmax_threshold_entire_df(df=master_ood_df.copy())
    # master_ood_df = evaluate_thresholds(df=master_ood_df.copy())
    # print_df(master_ood_df)
    #


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    folder_list = get_clean_folders(res_path)
    folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")

    main(result_path=res_path, folders=folder_list)
