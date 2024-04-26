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
        'depth': [3, 6, 9, 12],
        'fc_bool': [True, False],
        'fc_act': [True, False],
        'fc_batch': [True, False]
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
        print(params)
        parameters = copy.deepcopy(config_parameters)
        parameters.update(params)
        parameters['run_name'] = f"model_exhaustive_{i}"

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
