import argparse
import os
import random

import mlflow
import numpy as np
import torch
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


def set_run_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can be set to True for performance


def main():
    experiment = "bagging_ensemble_testing"
    #experiment = "bagging_ensemble_final"
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
    save_path: str = parameters.pop("save_path")
    model_name: str = parameters.get("classifier_name")
    prediction: str = parameters.pop("prediction")
    dataset_version: int = parameters.pop("dataset_version")
    eeg_epochs = parameters.get('eeg_epochs')
    overlapping_epochs: bool = parameters.pop("epoch_overlap", False)
    num_seconds: int = parameters.pop("num_seconds")
    use_age: bool = parameters.pop("use_age")
    age_scaling: str = parameters.pop("age_scaling")
    learning_rate: float = parameters.pop("learning_rate")
    batch_size: int = parameters.pop("batch_size")
    train_epochs: int = parameters.pop("training_epochs")
    earlystopping: int = parameters.pop("earlystopping")

    #########################################################################################################
    # Fixed parameters
    #########################################################################################################
    age_noise_prob = 0.75
    age_noise_level = 0.05
    augmentations = ['timereverse', 'smoothtimemask']
    other_parameters = {'smoothtimemask': {'mask_len_samples': 20}}
    augmentation_prob = 0.5

    #########################################################################################################
    # Bagging parameters
    #########################################################################################################
    depth = 6
    cnn_units = 11
    max_kernel_size = 45
    num_fc_layers = 2
    neurons_fc = 32
    use_batch_fc = True
    use_dropout_fc = True
    dropout_rate_fc = 0.4
    bagging_size = 0.8
    #########################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bagging_ensembles = 5
    experiment_path, folder_name = setup_experiment_path(save_path=save_path, config_path=config_path,
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

    #########################################################################################################
    # Loss function
    #########################################################################################################
    weight_tensor = dataset.get_class_weights(subjects=train_subjects, normalize=True)
    weight_tensor = weight_tensor.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    #########################################################################################################

    #########################################################################################################
    # Generators and Loaders
    #########################################################################################################
    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=augmentation_prob,
                                                **other_parameters)
    else:
        train_augmentations = []

    train_loader_list = []
    for i in range(num_bagging_ensembles):
        bag_seed = 42 * i
        set_run_seed(seed=bag_seed)
        subsample = random.sample(train_subjects, int(bagging_size * len(train_subjects)))
        train_gen = CauDataGenerator(subjects=subsample, dataset=dataset, device=device, split="train",
                                     use_age=use_age, augmentations=train_augmentations,
                                     age_noise_prob=age_noise_prob, age_noise_level=age_noise_level)
        train_loader_list.append(DataLoader(train_gen, batch_size=batch_size, shuffle=True))

    val_gen = CauDataGenerator(subjects=val_subjects, dataset=dataset, device=device, split="val", use_age=use_age)
    test_gen = CauDataGenerator(subjects=test_subjects, dataset=dataset, device=device, split="test", use_age=use_age)

    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False)

    #########################################################################################################
    # Run experiment
    #########################################################################################################
    with mlflow.start_run(run_name=folder_name):
        # Setup MLFLOW experiment
        classifiers = []

        for run_id in range(num_bagging_ensembles):
            train_loader = train_loader_list[run_id]

            # Introduce a seed for model-specifics
            model_seed = 42 + run_id
            set_run_seed(seed=model_seed)

            # Setting depth and cnn units to half of the standard to have simpler base models
            mlflow.start_run(run_name=f"{experiment}_{str(run_id)}", nested=True)
            run_path = create_run_folder(path=experiment_path, index=str(run_id))
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
            param.update(hyperparameters)
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
