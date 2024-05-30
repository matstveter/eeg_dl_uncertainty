import argparse
import os

from eegDlUncertainty.experiments.SingleModelExperiment import MCDExperiment, SWAGExperiment, SingleModelExperiment
from eegDlUncertainty.experiments.utils_exp import get_baseparameters_from_config


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
    parameters['run_name'] = args.run_name

    # exp = SingleModelExperiment(**parameters)
    # exp.run()

    exp = MCDExperiment(**parameters)
    exp.run()

    # exp = SWAGExperiment(**parameters)
    # exp.run()


if __name__ == "__main__":
    main()
