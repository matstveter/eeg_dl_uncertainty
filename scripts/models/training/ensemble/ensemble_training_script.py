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
from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.data.results.plotter import Plotter
from eegDlUncertainty.data.results.result_utils import calculate_ensemble_performance
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


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

    # TODO Remeber to remove this before experiments!!!
    # args.config_path = "conf.json"

    with open(os.path.join(os.path.dirname(__file__), "config_files", args.config_path)) as json_file:
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
    prediction_type = config['data']['prediction_type']
    which_one_vs_all = config['data']['which_one_vs_all_class']
    pairwise_class = config['data']['pairwise']

    ####################
    # Model Parameters #
    ####################
    model_name = config['model']['name']
    n_ensembles = config['model']['num_ensembles']

    if len(model_name) == 1 and (n_ensembles == 1 or n_ensembles == 0):
        raise ValueError("In ensemble training, only one model is defined, and number of ensembles is 1")

    if len(model_name) == 1:
        ensemble_models = model_name * n_ensembles
    else:
        ensemble_models = model_name

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
    ensembl_main_path = os.path.join(result_folder, f"ensemble_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}")
    os.mkdir(ensembl_main_path)
    shutil.copy(src=os.path.join(os.path.dirname(__file__), "config_files", args.config_path),
                dst=os.path.join(ensembl_main_path, args.config_path.split("/")[-1]))

    # Get data
    data = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                         epochs=eeg_epochs, prediction_type=prediction_type, which_one_vs_all=which_one_vs_all,
                         pairwise=pairwise_class)
    train_subjects, val_subjects, test_subjects = data.get_splits()

    # Set up the training data generator and loader
    train_gen = CauDataGenerator(subjects=train_subjects, dataset=data, device=device)
    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=0.2,
                                                random_state=meaning_of_life)
        # noinspection PyTypeChecker
        train_loader = AugmentedDataLoader(dataset=train_gen, transforms=train_augmentations, device=device,
                                           batch_size=batch_size,
                                           shuffle=True)
    else:
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True)

    # Set up the validation data generator and loader
    val_gen = CauDataGenerator(subjects=val_subjects, dataset=data, device=device)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    test_gen = CauDataGenerator(subjects=test_subjects, dataset=data, device=device)
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=True)

    if data.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    test_hist_list = []

    for i, model_name in enumerate(ensemble_models):
        print(f"\n**************************** MODEL {i+1} / {len(ensemble_models)} ****************************\n")
        model_path = os.path.join(ensembl_main_path, f"{model_name}_{i}")
        os.mkdir(model_path)

        # ------------------------------------------------------
        # Define model, optimiser, loss, and how to save results
        # ------------------------------------------------------
        hyperparameters = {"in_channels": data.num_channels, "num_classes": data.num_classes, "name": model_name,
                           "time_steps": data.eeg_len, "depth": 6, "save_path": model_path, "lr": learning_rate}
        train_hist = History(num_classes=data.num_classes, set_name="train", loader_lenght=len(train_loader),
                             save_path=model_path)
        val_hist = History(num_classes=data.num_classes, set_name="val", loader_lenght=len(val_loader),
                           save_path=model_path)
        test_hist = History(num_classes=data.num_classes, set_name="test", loader_lenght=len(test_loader),
                            save_path=model_path)
        test_hist_best_model = History(num_classes=data.num_classes, set_name="test", loader_lenght=len(test_loader),
                            save_path=model_path)

        model = MainClassifier(classifier_name=model_name, **hyperparameters)
        model.fit_model(train_loader=train_loader, val_loader=val_loader, training_epochs=training_epochs,
                        device=device, loss_fn=criterion, train_hist=train_hist,
                        val_history=val_hist)
        model.test_model(test_loader=test_loader, test_hist=test_hist, device=device, loss_fn=criterion)

        train_hist.save_to_pickle()
        val_hist.save_to_pickle()

        best_model = MainClassifier(classifier_name=model_name,
                                    pretrained=model.model_path(with_ext=True),
                                    **hyperparameters)
        best_model.test_model(test_loader=test_loader, test_hist=test_hist_best_model, device=device, loss_fn=criterion)

        # Save the data
        test_hist.save_to_pickle()

        plot = Plotter(train_dict=train_hist.get_as_dict(),
                       val_dict=val_hist.get_as_dict(),
                       test_dict=test_hist.get_as_dict(),
                       test_dict_best_model=test_hist_best_model.get_as_dict(),
                       save_path=model_path)
        plot.produce_plots()
        test_hist_list.append(test_hist_best_model)

    calculate_ensemble_performance(test_hist_list, path=ensembl_main_path)


if __name__ == "__main__":
    main()
