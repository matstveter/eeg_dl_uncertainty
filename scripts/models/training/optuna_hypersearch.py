import argparse
import os
import random
from typing import List, Optional, Union
import mlflow
import numpy
import numpy as np
import torch
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader
import optuna

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History, get_history_objects
from eegDlUncertainty.data.results.utils_mlflow import add_config_information
from eegDlUncertainty.experiments.utils_exp import cleanup_function, create_run_folder, get_parameters_from_config, \
    prepare_experiment_environment, \
    setup_experiment_path
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


def objective(trial, fixed_params):
    # Fixed parameters
    dataset_version = fixed_params["dataset_version"]
    prediction = fixed_params["prediction"]
    use_age = fixed_params["use_age"]
    save_path = fixed_params["save_path"]
    model_name = fixed_params["model_name"]
    use_test_set = fixed_params["use_test_set"]
    train_epochs = fixed_params["train_epochs"]
    batch_size = fixed_params["batch_size"]
    learning_rate = fixed_params["learning_rate"]
    earlystopping = fixed_params["earlystopping"]
    augmentations = fixed_params["augmentations"]
    augmentation_prob = fixed_params["augmentation_prob"]
    config_path = fixed_params.get("config_path")
    overlapping_epochs = fixed_params.get("overlapping_epochs", False)
    direction = fixed_params.get("direction")

    # Define random state and device
    random_state: int = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)

    mlflow.start_run(run_name=f"run_{str(trial.number)}", nested=True)
    run_path = create_run_folder(path=save_path, index=str(trial.number))

    # Define the hyperparameters to search
    age_scaling = trial.suggest_categorical("age_scaling", ["min_max", "standard"])
    cnn_units = trial.suggest_categorical("cnn_units", [30, 50, 70, 90, 110])
    depth = trial.suggest_int("depth", 1, 12)
    eeg_epochs = trial.suggest_categorical("eeg_epochs", ["all", "spread"])
    num_seconds = trial.suggest_categorical("num_seconds", [5, 10, 20, 30])

    # Create a dictionary of the parameters
    params = {
        "age_scaling": age_scaling,
        "cnn_units": cnn_units,
        "depth": depth,
        "eeg_epochs": eeg_epochs,
        "num_seconds": num_seconds
    }
    mlflow.log_params(params)

    #########################################################################################################
    # Dataset
    #########################################################################################################
    dataset = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                            epochs=eeg_epochs, overlapping_epochs=overlapping_epochs, age_scaling=age_scaling,
                            save_dir=run_path)
    train_subjects, val_subjects, test_subjects = dataset.get_splits()
    if "test" in config_path:
        train_subjects = train_subjects[0:10]
        val_subjects = val_subjects[0:10]
        test_subjects = test_subjects[0:10]

    if dataset.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #########################################################################################################
    # Generators
    #########################################################################################################
    train_gen = CauDataGenerator(subjects=train_subjects, dataset=dataset, device=device, split="train",
                                 use_age=use_age)
    val_gen = CauDataGenerator(subjects=val_subjects, dataset=dataset, device=device, split="val",
                               use_age=use_age)
    test_gen = CauDataGenerator(subjects=test_subjects, dataset=dataset, device=device, split="test",
                                use_age=use_age)
    #########################################################################################################
    # Loaders
    #########################################################################################################
    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=augmentation_prob,
                                                random_state=random_state)
        # noinspection PyTypeChecker
        train_loader = AugmentedDataLoader(dataset=train_gen, transforms=train_augmentations, device=device,
                                           batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False)

    hyperparameters = {"in_channels": dataset.num_channels,
                       "num_classes": dataset.num_classes,
                       "time_steps": dataset.eeg_len,
                       "save_path": run_path,
                       "learning_rate": learning_rate}

    classifier = MainClassifier(model_name=model_name, **hyperparameters)
    train_history, val_history = get_history_objects(train_loader=train_loader, val_loader=val_loader,
                                                     save_path=run_path, num_classes=dataset.num_classes)
    try:
        classifier.fit_model(train_loader=train_loader, training_epochs=train_epochs, device=device,
                             loss_fn=criterion, earlystopping_patience=earlystopping,
                             val_loader=val_loader, train_hist=train_history, val_history=val_history)
    except torch.cuda.OutOfMemoryError as e:
        mlflow.set_tag("Exception", "CUDA Out of Memory Error")
        mlflow.log_param("Exception Message", str(e))
        cleanup_function(experiment_path=run_path)
        print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")

        if direction == "minimize":
            return np.inf
        else:
            return -np.inf

    else:
        if use_test_set:
            evaluation_history = History(num_classes=dataset.num_classes, set_name="test",
                                         loader_lenght=len(test_loader), save_path=run_path)
            classifier.test_model(test_loader=test_loader, device=device, test_hist=evaluation_history,
                                  loss_fn=criterion)
        else:
            evaluation_history = History(num_classes=dataset.num_classes, set_name="test_val",
                                         loader_lenght=len(val_loader), save_path=run_path)
            classifier.test_model(test_loader=val_loader, device=device, test_hist=evaluation_history,
                                  loss_fn=criterion)

        train_history.save_to_mlflow()
        train_history.save_to_pickle()
        val_history.save_to_mlflow()
        val_history.save_to_pickle()
        evaluation_history.save_to_mlflow()
        evaluation_history.save_to_pickle()

    finally:
        mlflow.end_run()

    x = trial.suggest_float("x", -10, 10)

    val = (x - 2) ** 2 + cnn_units
    mlflow.log_metric("val", val)
    mlflow.end_run()
    return val


def main():
    #########################################################################################################
    # Get arguments and read config file
    #########################################################################################################
    arg_parser = argparse.ArgumentParser(description="Run script for training a model")
    arg_parser.add_argument("-c", "--config_path", type=str, help="Path to config (.json) file",
                            required=False, default=None)
    arg_parser.add_argument("--run_name", type=str, help="Run name for MLFlow", default=None)
    args = arg_parser.parse_args()
    if args.config_path is None:
        args.config_path = "test_conf.json"
        print("WARNING!!!! No config argument added, using the first conf.json file, mostly used for pycharm!")

    config_path = os.path.join(os.path.dirname(__file__), "config_files", args.config_path)
    parameters = get_parameters_from_config(config_path=config_path)

    #########################################################################################################
    # Init variables from config file
    #########################################################################################################
    use_test_set: bool = parameters.pop("use_test_set", False)
    save_path: str = parameters.pop("save_path")
    model_name: str = parameters.get("classifier_name")

    experiment_path, folder_name = setup_experiment_path(save_path=save_path,
                                                         config_path=config_path,
                                                         experiment=model_name)

    fixed_params = {
        "config_path": config_path,
        "use_test_set": use_test_set,
        "save_path": experiment_path,
        "model_name": model_name,
        "dataset_version": parameters.get("dataset_version"),
        "prediction": parameters.get("prediction"),
        "use_age": parameters.get("use_age"),
        "train_epochs": parameters.get("training_epochs"),
        "batch_size": parameters.get("batch_size"),
        "learning_rate": parameters.get("learning_rate"),
        "earlystopping": parameters.get("earlystopping"),
        "augmentations": parameters.get("augmentations"),
        "augmentation_prob": parameters.get("augmentation_prob"),
        "direction": "minimize"
    }

    if fixed_params['direction'] not in ("minimize", "maximize"):
        raise ValueError("direction must be either 'minimize' or 'maximize'")

    experiment_name = "optuna_search"
    prepare_experiment_environment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=experiment_name):
        study = optuna.create_study(direction=fixed_params["direction"])
        study.optimize(lambda trial: objective(trial, fixed_params), n_trials=10)

        print(study.best_params)

        for k, v in study.best_params.items():
            print(f"{k}: {v}")
            mlflow.log_param(k, v)







if __name__ == "__main__":
    main()
