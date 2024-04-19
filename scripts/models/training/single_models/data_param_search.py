import argparse
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_random_hyperparameters():
    random.seed()
    # Define your ranges or sets of possible values for each hyperparameter
    parameters = {
        'use_age': random.choice(['True', 'False']),
        'age_scaling': random.choice(['standard', 'min_max']),
        'mc_dropout_enabled': [True, False],
        'mc_dropout_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    }

    return parameters


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

    for i in range(num_random_search_iterations):
        parameters = {}
        if args.run_name is None:
            run_name = f"dataparam_search_{i}"
        else:
            run_name = f"{args.run_name}_{i}"

        parameters = get_baseparameters_from_config(config_path=config_path)
        parameters['config_path'] = config_path
        parameters['run_name'] = run_name

        hyper_param = generate_random_hyperparameters()
        parameters.update(hyper_param)

        # parameters['experiment_name'] = f"HyperParamExp_{parameters['classifier_name']}"

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
