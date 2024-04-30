import argparse
import itertools
import os

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_grid_hyperparameters():
    param_grid = {
        'num_seconds': [5, 10, 20, 30, 60],
        'epochs': [1, 2, 4, 6, 8, 10],
    }
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
    parameters = get_baseparameters_from_config(config_path=config_path)
    parameters['config_path'] = config_path
    parameters['experiment_name'] = "hyperparameter_search"

    for i, params in enumerate(generate_grid_hyperparameters()):
        parameters['run_name'] = f"data_info_{i}"
        parameters.update(params)
        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
