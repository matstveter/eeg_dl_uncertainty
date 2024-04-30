import argparse
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_random_hyperparameters():
    random.seed()
    # Package the parameters into a dictionary
    parameters = {
        'batch_size': random.choice([32, 64, 128]),
        'learning_rate': random.choice([0.05, 0.01, 0.001, 0.0001]),
        'num_seconds': random.choice([20, 30, 60]),
        'depth': random.choice([2, 3, 6, 9, 12]),
        'eeg_epochs': random.choice([1, 2, 3, 4, 5]),
        'epoch_overlap': random.choice([True, False]),
        'mc_dropout_enabled': random.choice([True, False]),
        'classifier_name': random.choice(['InceptionNetwork', 'InceptionWide']),
        'age_scaling': random.choice(['standard', 'min_max']),
        'dataset_version': random.choice([1, 2, 3, 4, 5, 6, 7, 8])
    }

    if parameters['mc_dropout_enabled']:
        parameters['mc_dropout_rate'] = random.choice([0.2, 0.3, 0.4, 0.5])

    return parameters


def main():
    num_random_search_iterations = 500

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
            run_name = f"hyperparam_search_{i}"
        else:
            run_name = f"{args.run_name}_{i}"

        parameters = get_baseparameters_from_config(config_path=config_path)
        parameters['config_path'] = config_path
        parameters['run_name'] = run_name
        parameters['experiment_name'] = "hyperparameter_search"

        hyper_param = generate_random_hyperparameters()
        parameters.update(hyper_param)

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
