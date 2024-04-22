import argparse
import os
import random

import numpy as np

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def main():
    num_random_search_iterations = 250

    # Argumentparser
    arg_parser = argparse.ArgumentParser(description="Run script for training a model")
    arg_parser.add_argument("-c", "--config_path", type=str, help="Path to config (.json) file",
                            required=False, default=None)
    arg_parser.add_argument("--run_name", type=str, help="Run name for MLFlow", default=None)
    args = arg_parser.parse_args()
    if args.config_path is None:
        args.config_path = "test_conf.json"
        print("WARNING!!!! No config argument added, using the first conf.json file, mostly used for pycharm!")

    config_path = os.path.join(os.path.dirname(__file__), "config_files", args.config_path)

    dataset_version = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset_version = np.repeat(dataset_version, 5)
    parameters = get_baseparameters_from_config(config_path=config_path)
    parameters['config_path'] = config_path

    for d_v in dataset_version:
        parameters['run_name'] = f"dataset_version_{d_v}"
        parameters['dataset_version'] = d_v
        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
