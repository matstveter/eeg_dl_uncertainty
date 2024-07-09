import argparse
import os
import random
from typing import List, Optional, Union

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
from eegDlUncertainty.experiments.utils_exp import cleanup_function, create_run_folder, get_parameters_from_config, \
    prepare_experiment_environment, \
    setup_experiment_path
from eegDlUncertainty.models.classifiers.ensemble import Ensemble
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


def main():
    experiment = "depth_ensemble"
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

    # Augmentation related information
    augmentations: List[Union[str, None]] = parameters.pop("augmentations")
    augmentation_prob: Optional[float] = parameters.pop("augmentation_prob", 0.2)

    # Model training
    model_name: str = parameters.get("classifier_name")
    train_epochs: int = parameters.pop("training_epochs")
    batch_size: int = parameters.pop("batch_size")
    learning_rate: float = parameters.pop("learning_rate")
    earlystopping: int = parameters.pop("earlystopping")


    random_state: int = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)

    experiment_path, folder_name = setup_experiment_path(save_path=save_path,
                                                         config_path=config_path,
                                                         experiment=experiment)
    experiment_name = f"{experiment}_experiments"
    prepare_experiment_environment(experiment_name=experiment_name)
    #########################################################################################################
    # Dataset
    #########################################################################################################
    dataset = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                            epochs=eeg_epochs, overlapping_epochs=overlapping_epochs, age_scaling=age_scaling)
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

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=folder_name):
        # Setup MLFLOW experiment
        depths = [3, 5, 6, 9, 10, 12]
        classifiers = []

        for run_id in range(len(depths)):

            mlflow.start_run(run_name=f"{experiment}_{str(run_id)}", nested=True)
            run_path = create_run_folder(path=experiment_path, index=str(run_id))
            hyperparameters = {"in_channels": dataset.num_channels,
                               "num_classes": dataset.num_classes,
                               "time_steps": dataset.eeg_len,
                               "save_path": run_path,
                               "learning_rate": learning_rate}
            param.update(hyperparameters)
            param['depth'] = depths[run_id]

            add_config_information(config=param, dataset="CAUEEG")

            classifier = MainClassifier(model_name=model_name, **hyperparameters)
            train_history, val_history = get_history_objects(train_loader=train_loader, val_loader=val_loader,
                                                             save_path=save_path, num_classes=dataset.num_classes)
            try:
                _ = classifier.fit_model(train_loader=train_loader, training_epochs=train_epochs, device=device,
                                         loss_fn=criterion, earlystopping_patience=earlystopping,
                                         val_loader=val_loader, train_hist=train_history, val_history=val_history)
            except torch.cuda.OutOfMemoryError as e:
                mlflow.set_tag("Exception", "CUDA Out of Memory Error")
                mlflow.log_param("Exception Message", str(e))
                cleanup_function(experiment_path=experiment_path)
                print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
                break
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

                classifiers.append(classifier)

            finally:
                mlflow.end_run()

        ens = Ensemble(classifiers=classifiers, device=device)

        if use_test_set:
            ens.ensemble_performance_and_uncertainty(data_loader=test_loader, device=device, save_path=run_path,
                                                     save_to_mlflow=True, save_to_pickle=True,
                                                     save_name="ensemble_results_test")
            eval_dataset_shifts(ensemble_class=ens, test_subjects=test_subjects, dataset=dataset,
                                device=device, use_age=use_age, batch_size=batch_size,
                                save_path=run_path)
        else:
            ens.ensemble_performance_and_uncertainty(data_loader=val_loader, device=device, save_path=run_path,
                                                     save_to_mlflow=True, save_to_pickle=True,
                                                     save_name="ensemble_results_val")
            eval_dataset_shifts(ensemble_class=ens, test_subjects=val_subjects, dataset=dataset,
                                device=device, use_age=use_age, batch_size=batch_size,
                                save_path=run_path)

        # ood_results = ood_experiment(classifiers, dataset_version=dataset_version, num_seconds=num_seconds,
        #                              age_scaling=age_scaling, device=device, batch_size=batch_size,
        #                              save_path=experiment_path)


if __name__ == "__main__":
    main()
