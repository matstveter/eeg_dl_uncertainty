import argparse
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_random_hyperparameters():
    # Define your ranges or sets of possible values for each hyperparameter
    batch_size = [16, 32, 64, 128, 256]  # Example range for cnn_units
    lr = [0.01, 0.001, 0.0001, 0.00001]
    num_seconds = range(9, 41, 2)
    depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    batch_size = random.choice(batch_size)
    lr = random.choice(lr)
    num_sec = random.choice(num_seconds)
    depth = random.choice(depth)

    # Package the parameters into a dictionary
    parameters = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'num_seconds': num_sec,
        'depth': depth
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
            run_name = f"hyperparam_search_{i}"
        else:
            run_name = f"{args.run_name}_{i}"

        parameters = get_baseparameters_from_config(config_path=config_path)
        parameters['config_path'] = config_path
        parameters['run_name'] = run_name

        hyper_param = generate_random_hyperparameters()
        print(hyper_param)
        parameters.update(hyper_param)

        # parameters['experiment_name'] = f"HyperParamExp_{parameters['classifier_name']}"

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
