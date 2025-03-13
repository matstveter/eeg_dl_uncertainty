import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from scripts.plots.utils import get_clean_folders, get_full_extension, get_remaining_folders_without_name, \
    get_ensemble_names, read_all_dataset_shifts, read_pkl_to_dict


def main(result_path, folders):
    ensemble_names = get_ensemble_names(folder_list=folders)
    full_paths = get_full_extension(res_path=result_path, folder_list=folders, ood=True)

    dataset_shifts = "ood_results.pkl"

    for path in full_paths:

        break


if __name__ == '__main__':
    res_path = "/home/tvetern/PhD/dl_uncertainty/results/"
    folder_list = get_clean_folders(res_path)
    folder_list = get_remaining_folders_without_name(folder_list=folder_list, name="NORMAL")

    main(result_path=res_path, folders=folder_list)
