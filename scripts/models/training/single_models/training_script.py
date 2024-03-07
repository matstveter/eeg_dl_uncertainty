import argparse
import json
import os
import random
import shutil
from datetime import datetime
import numpy
import torch
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset


def main():
    meaning_of_life = 42
    random.seed(meaning_of_life)
    numpy.random.seed(meaning_of_life)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argumentparser
    # Read argparse and config file
    arg_parser = argparse.ArgumentParser(description="Run script for training a model")
    arg_parser.add_argument("-c", "--config_path", type=str, help="Path to config (.json) file")
    args = arg_parser.parse_args()
    args.config_path = "conf.json"

    with open(os.path.join(os.path.dirname(__file__), "config_files",args.config_path)) as json_file:
        config = json.load(json_file)

    ###########################
    # Get dataset information #
    ###########################
    prediction = config['data']['prediction']
    if config['data']['overlap']:
        prediction = f"{prediction}-no-overlap"
    dataset_version = config['data']['version']
    eeg_epochs = config['data']['eeg_epochs']
    num_seconds = config['data']['num_seconds']

    ####################
    # Model Parameters #
    ####################
    model_name = config['model']['name']
    activation = config['model']['activation']

    ###################
    # Hyperparameters #
    ###################
    learning_rate = config['hyperparameters']['learning_rate']
    batch_size = config['hyperparameters']['batch_size']
    training_epochs = config['hyperparameters']['epochs']
    augmentations = config['hyperparameters']['augmentations']

    #################
    # Paths         #
    #################
    result_folder = config['result_folder']
    paths = os.path.join(result_folder, f"{model_name}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}")
    os.mkdir(paths)

    shutil.copy(src=os.path.join(os.path.dirname(__file__), "config_files", args.config_path),
                dst=os.path.join(paths, args.config_path.split("/")[-1]))

    # Get data
    data = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                         epochs=eeg_epochs)
    train_subjects, val_subjects, test_subjects = data.get_splits()

    train_dataset = CauDataGenerator(subjects=train_subjects, dataset=data, device=device)

    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=0.5,
                                                random_state=meaning_of_life)
        # noinspection PyTypeChecker
        train_gen = AugmentedDataLoader(dataset=train_dataset, transforms=train_augmentations, device=device,
                                        batch_size=batch_size,
                                        shuffle=True)
    else:
        train_gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Test Model

    # Save history and figures

    # save last model


if __name__ == "__main__":
    main()
