import argparse
import copy
import itertools
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_grid_hyperparameters():
    # Define the grid of parameters as lists of all possible values
    param_grid = {
        'cnn_units': [8, 16, 32, 64, 128],
        'depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        'max_kernel_size': [20, 40, 80],
        'batch_size': [8, 16, 32, 64, 128, 256]
        # 'mc_dropout_enabled': [True, False],
        # 'mc_dropout_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    }

    # Use itertools.product to generate all possible combinations of these parameters
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def main():
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

    config_parameters = get_baseparameters_from_config(config_path=config_path)
    config_parameters['config_path'] = config_path

    for i, params in enumerate(generate_grid_hyperparameters()):
        parameters = copy.deepcopy(config_parameters)
        parameters.update(params)
        parameters['run_name'] = f"model_exhaustive_{i}"

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
