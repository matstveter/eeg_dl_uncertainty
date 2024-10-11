import argparse
import os
import random
from typing import List

import mlflow
import numpy
import numpy as np
import optuna
import torch
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History, get_history_objects
from eegDlUncertainty.experiments.utils_exp import cleanup_function, create_run_folder, get_parameters_from_config, \
    prepare_experiment_environment, \
    setup_experiment_path
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


def objective(trial, fixed_params):
    print("Running experiment number: ", trial.number)
    # Fixed parameters
    dataset_version: int = fixed_params["dataset_version"]
    prediction: str = fixed_params["prediction"]
    use_age: bool = fixed_params["use_age"]
    save_path: str = fixed_params["save_path"]
    model_name: str = fixed_params["model_name"]
    use_test_set: bool = fixed_params["use_test_set"]
    train_epochs: int = fixed_params["train_epochs"]
    batch_size: int = fixed_params["batch_size"]
    learning_rate: float = fixed_params["learning_rate"]
    earlystopping: int = fixed_params["earlystopping"]
    config_path: str = fixed_params.get("config_path")
    overlapping_epochs: bool = fixed_params.get("overlapping_epochs", False)
    metric: str = fixed_params.get("metric")
    augmentations = fixed_params.get("augmentations")
    augmentation_prob = fixed_params.get("augmentation_prob")
    eeg_epochs = fixed_params.get("eeg_epochs")
    age_scaling = fixed_params.get("age_scaling")
    num_seconds = fixed_params.get("num_seconds")
    depth = fixed_params.get("depth")
    cnn_units = fixed_params.get("cnn_units")
    max_kernel_size = fixed_params.get("max_kernel_size")
    num_fc_layers = fixed_params.get("num_fc_layers")
    neurons_fc = fixed_params.get("neurons_fc")
    use_batch_fc = fixed_params.get("use_batch_fc")
    use_dropout_fc = fixed_params.get("use_dropout_fc")
    dropout_rate_fc = fixed_params.get("dropout_rate_fc")

    # Define random state and device
    random_state: int = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)

    # Start mlflow run and create a folder for the current run
    mlflow.start_run(run_name=f"run_{str(trial.number)}", nested=True)
    run_path = create_run_folder(path=save_path, index=str(trial.number))

    params = {}

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
                       "learning_rate": learning_rate,
                       "cnn_units": cnn_units,
                       "depth": depth,
                       "max_kernel_size": max_kernel_size,
                       "num_fc_layers": num_fc_layers,
                       "neurons_fc": neurons_fc,
                       "use_batch_fc": use_batch_fc,
                       "use_dropout_fc": use_dropout_fc,
                       "dropout_rate_fc": dropout_rate_fc,
                       }

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

        if metric == "loss":
            return np.inf
        else:
            # If the metric is not loss, we return 0, like for accuracy, auc
            return 0

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

        optimizing_metric = evaluation_history.get_history_metric(metric_name=metric)[0]
        mlflow.log_metric(f"optimizing_metric_{metric}", optimizing_metric)
        return optimizing_metric
    finally:
        mlflow.end_run()


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

    metric: str = "loss"

    if metric not in ("loss", "accuracy", "auc", "mcc"):
        raise ValueError("metric must be either 'loss', 'accuracy', 'auc', 'mcc'!")

    if metric == "loss":
        direction = "minimize"
    else:
        direction = "maximize"

    experiment_path, folder_name = setup_experiment_path(save_path=save_path,
                                                         config_path=config_path,
                                                         experiment=model_name)

    fixed_params = {
        "config_path": config_path,
        "use_test_set": use_test_set,
        "save_path": experiment_path,
        "model_name": model_name,
        "direction": direction,
        "metric": metric,
        "prediction": parameters.get("prediction"),
        "dataset_version": parameters.get("dataset_version"),
        "eeg_epochs": parameters.get("eeg_epochs"),
        "overlapping_epochs": parameters.get("overlapping_epochs"),
        "use_age": parameters.get("use_age"),
        "age_scaling": parameters.get("age_scaling"),
        "num_seconds": parameters.get("num_seconds"),
        "train_epochs": parameters.get("training_epochs"),
        "batch_size": parameters.get("batch_size"),
        "learning_rate": parameters.get("learning_rate"),
        "earlystopping": parameters.get("earlystopping"),
        "augmentations": parameters.get("augmentations"),
        "augmentation_prob": parameters.get("augmentation_prob"),
        "depth": parameters.get("depth"),
        "cnn_units": parameters.get("cnn_units"),
        "max_kernel_size": parameters.get("max_kernel_size"),
        "num_fc_layers": parameters.get("num_fc_layers"),
        "neurons_fc": parameters.get("neurons_fc"),
        "use_batch_fc": parameters.get("use_batch_fc"),
        "use_dropout_fc": parameters.get("use_dropout_fc"),
        "dropout_rate_fc": parameters.get("dropout_rate_fc"),
    }
    experiment_name = "optuna_031024"
    prepare_experiment_environment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=experiment_name):
        study = optuna.create_study(study_name="hyper_search", direction=fixed_params["direction"])
        # Optimization
        study.optimize(lambda trial: objective(trial, fixed_params), n_trials=250)

        # Log the best parameters
        for k, v in study.best_params.items():
            mlflow.log_param(k, v)


if __name__ == "__main__":
    main()
