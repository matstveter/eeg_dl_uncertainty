import argparse
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_random_hyperparameters(model_name):
    random.seed()
    if model_name == "InceptionNetwork":
        # Package the parameters into a dictionary
        params = {
            'cnn_units': random.choice(range(8, 64, 2)),
            'depth': random.choice([3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
            'max_kernel_size': random.choice([20, 40, 60, 80, 120]),
            'batch_size': random.choice([2, 4, 8, 16, 32, 64, 128, 256]),
            'mc_dropout_enabled': random.choice([True, False]),
            'mc_dropout_rate': random.choice([0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]),
            'swa_enabled': random.choice([True, False])
        }
    else:
        raise KeyError(f"Unrecognized model name : {model_name}")
    return params


def main():
    num_random_search_iterations = 150

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
            run_name = f"model_exploit_{i}"
        else:
            run_name = f"{args.run_name}_{i}"

        parameters = get_baseparameters_from_config(config_path=config_path)
        parameters['config_path'] = config_path
        parameters['run_name'] = run_name
        parameters['experiment_name'] = "model_search"

        # parameters['experiment_name'] = f"ModelExploitation_{parameters['classifier_name']}"

        if args.config_path != "test_conf.json":
            model_param = generate_random_hyperparameters(model_name=parameters['classifier_name'])
            parameters.update(model_param)

        exp = SingleModelExperiment(**parameters)
        exp.run()

        if args.config_path == "test_conf.json":
            break


if __name__ == "__main__":
    main()
