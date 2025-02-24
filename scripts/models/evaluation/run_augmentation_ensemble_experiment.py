import argparse
import os
import random

import mlflow
import numpy
import torch
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History, get_history_objects
from eegDlUncertainty.data.results.utils_mlflow import add_config_information
from eegDlUncertainty.experiments.dataset_shift_experiment import eval_dataset_shifts
from eegDlUncertainty.experiments.ood_experiments import ood_exp
from eegDlUncertainty.experiments.utils_exp import cleanup_function, create_run_folder, get_parameters_from_config, \
    prepare_experiment_environment, \
    setup_experiment_path
from eegDlUncertainty.models.classifiers.ensemble import Ensemble
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


def main():
    experiment = "augmentation_ensemble_final"
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
    param = parameters.copy()
    use_test_set: bool = parameters.pop("use_test_set", False)
    save_path: str = parameters.pop("save_path")
    run_name: str = args.run_name

    # Data related variables
    dataset_version: int = parameters.pop("dataset_version")
    prediction: str = parameters.pop("prediction")
    use_age: bool = parameters.pop("use_age")
    age_scaling: str = parameters.pop("age_scaling")
    num_seconds: int = parameters.pop("num_seconds")
    eeg_epochs: str = parameters.pop("eeg_epochs")
    overlapping_epochs: bool = parameters.pop("epoch_overlap", False)

    # Model training
    model_name: str = parameters.get("classifier_name")
    train_epochs: int = parameters.pop("training_epochs")
    batch_size: int = parameters.pop("batch_size")
    learning_rate: float = parameters.pop("learning_rate")
    earlystopping: int = parameters.pop("earlystopping")

    # General variables
    model_p = {
        'depth': parameters.pop("depth"),
        'cnn_units': parameters.pop("cnn_units"),
        'max_kernel_size': parameters.pop("max_kernel_size")
    }

    random_state: int = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)

    # Experiment specific variables
    augmentation_prob = parameters.pop("augmentation_prob", 0.5)

    experiment_path, folder_name = setup_experiment_path(save_path=save_path,
                                                         config_path=config_path,
                                                         experiment=experiment)
    experiment_name = f"{experiment}_experiments"
    prepare_experiment_environment(experiment_name=experiment_name)
    #########################################################################################################
    # Dataset
    #########################################################################################################
    dataset = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                            epochs=eeg_epochs, overlapping_epochs=overlapping_epochs, age_scaling=age_scaling,
                            save_dir=experiment_path)
    train_subjects, val_subjects, test_subjects = dataset.get_splits()

    if "test" in config_path:
        train_subjects = train_subjects[0:100]
        val_subjects = val_subjects[0:25]
        test_subjects = test_subjects[0:20]

    if dataset.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #########################################################################################################
    # Generators
    #########################################################################################################
    train_gen = CauDataGenerator(subjects=train_subjects, dataset=dataset, device=device, split="train",
                                 use_age=use_age)
    val_gen = CauDataGenerator(subjects=val_subjects, dataset=dataset, device=device, split="val", use_age=use_age)
    test_gen = CauDataGenerator(subjects=test_subjects, dataset=dataset, device=device, split="test", use_age=use_age)

    #########################################################################################################
    # Loaders
    #########################################################################################################
    train_loader_list = []
    num_augmentations = ['timereverse', 'signflip', 'ftsurrogate', 'channelsdropout', 'smoothtimemask']

    other_args = {'timereverse': {},
                  'signflip': {},
                  'ftsurrogate': {'phase_noise_magnitude': 0.1, 'channel_indep': True},
                  'channelsdropout': {'p_drop': 0.3},
                  'smoothtimemask': {'max_len_samples': 15}
                  }

    for aug in num_augmentations:
        train_augmentations = get_augmentations(aug_names=[aug], probability=augmentation_prob,
                                                random_state=random_state, **other_args)
        train_loader_list.append(AugmentedDataLoader(dataset=train_gen, transforms=train_augmentations, device=device,
                                                     batch_size=batch_size, shuffle=True))

    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False)

    #########################################################################################################
    # Run experiment
    #########################################################################################################

    with mlflow.start_run(run_name=folder_name):
        # Setup MLFLOW experiment
        classifiers = []

        for run_id in range(len(train_loader_list)):
            # Get train loader for current run
            train_loader = train_loader_list[run_id]
            # Setup MLFLOW run
            mlflow.start_run(run_name=f"{experiment}_{str(run_id)}", nested=True)
            # Create run folder
            run_path = create_run_folder(path=experiment_path, index=str(run_id))
            # Create a model specific hyperparameter dictionary
            hyperparameters = {"in_channels": dataset.num_channels,
                               "num_classes": dataset.num_classes,
                               "time_steps": dataset.eeg_len,
                               "save_path": run_path,
                               "learning_rate": learning_rate}
            hyperparameters.update(model_p)
            # Update the hyperparameter dictionary with the general hyperparameters
            param.update(hyperparameters)
            # Add the hyperparameters to the MLFLOW run
            add_config_information(config=param, dataset="CAUEEG")
            # Create a classifier object
            classifier = MainClassifier(model_name=model_name, **hyperparameters)
            # Get history objects
            train_history, val_history = get_history_objects(train_loader=train_loader, val_loader=val_loader,
                                                             save_path=save_path, num_classes=dataset.num_classes)
            # Try to train the model, catch CUDA out of memory error, mostly used during hyperparameter search
            try:
                # Fit the model
                classifier.fit_model(train_loader=train_loader, training_epochs=train_epochs, device=device,
                                     loss_fn=criterion, earlystopping_patience=earlystopping,
                                     val_loader=val_loader, train_hist=train_history, val_history=val_history)
            except torch.cuda.OutOfMemoryError as e:
                # Log the error message and cleanup the experiment
                mlflow.set_tag("Exception", "CUDA Out of Memory Error")
                mlflow.log_param("Exception Message", str(e))
                # This function deletes the run folder and the MLFLOW run
                cleanup_function(experiment_path=experiment_path)
                print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
                break
            else:
                evaluation_history_val = History(num_classes=dataset.num_classes, set_name="test_val",
                                                 loader_lenght=len(val_loader), save_path=run_path)
                classifier.test_model(test_loader=val_loader, device=device, test_hist=evaluation_history_val,
                                      loss_fn=criterion)

                evaluation_history_test = History(num_classes=dataset.num_classes, set_name="test",
                                                  loader_lenght=len(test_loader), save_path=run_path)
                classifier.test_model(test_loader=test_loader, device=device, test_hist=evaluation_history_test,
                                      loss_fn=criterion)

                train_history.save_to_mlflow()
                train_history.save_to_pickle()
                val_history.save_to_mlflow()
                val_history.save_to_pickle()
                evaluation_history_val.save_to_mlflow()
                evaluation_history_val.save_to_pickle()
                evaluation_history_test.save_to_mlflow()
                evaluation_history_test.save_to_pickle()

                classifiers.append(classifier)

            finally:
                mlflow.end_run()

        # Initialize ensemble model with the trained classifiers
        ens = Ensemble(classifiers=classifiers, device=device)
        # Set the temperature scale for the ensemble
        ens.set_temperature_scale_ensemble(data_loader=val_loader, device=device, criterion=criterion)
        # Test the ensemble model on the validation and test set
        ens.ensemble_performance_and_uncertainty(data_loader=val_loader, device=device, save_path=run_path,
                                                 save_to_mlflow=True, save_to_pickle=True,
                                                 save_name="ensemble_results_val")
        ens.ensemble_performance_and_uncertainty(data_loader=test_loader, device=device, save_path=run_path,
                                                 save_to_mlflow=True, save_to_pickle=True,
                                                 save_name="ensemble_results_test")
        # Evaluate the dataset shifts on the ensemble model using the test set
        eval_dataset_shifts(ensemble_class=ens, test_subjects=test_subjects, dataset=dataset,
                            device=device, use_age=use_age, batch_size=batch_size,
                            save_path=run_path)
        # Run the OOD experiment
        ood_exp(ensemble_class=ens, dataset_version=dataset_version,
                num_seconds=num_seconds,
                age_scaling=age_scaling, device=device, batch_size=batch_size,
                save_path=experiment_path)


if __name__ == "__main__":
    main()
