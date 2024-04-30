import argparse
import os
import random

from eegDlUncertainty.experiments.SingleModelExperiment import SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


def generate_random_hyperparameters():
    random.seed()
    possible_augmentations = ['gaussiannoise', 'timereverse', 'signflip', 'ftsurrogate', 'channelsshuffle',
                              'channelsdropout', 'smoothtimemask', 'bandstopfilter']

    num_to_select = random.randint(2, len(possible_augmentations))
    return random.sample(possible_augmentations, num_to_select)


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
            run_name = f"random_search_augmentations_{i}"
        else:
            run_name = f"{args.run_name}_{i}"

        parameters = get_baseparameters_from_config(config_path=config_path)
        parameters['config_path'] = config_path
        parameters['run_name'] = run_name

        parameters['experiment_name'] = f"augmentation_explorer"

        augmentation_param = generate_random_hyperparameters()
        parameters['augmentations'] = augmentation_param

        exp = SingleModelExperiment(**parameters)
        exp.run()


if __name__ == "__main__":
    main()
