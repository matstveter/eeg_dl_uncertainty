import argparse
import os

import mlflow
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
from eegDlUncertainty.models.classifiers.main_classifier import FGEClassifier


def main():
    # experiment = "FGE_ensemble_final"
    experiment = "testing"
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
    # Normal parameters
    #########################################################################################################
    depth: int = parameters.get("depth")
    cnn_units = parameters.get("cnn_units")
    max_kernel_size = parameters.get("max_kernel_size")
    num_fc_layers = parameters.get("num_fc_layers")
    neurons_fc = parameters.get("neurons_fc")
    use_batch_fc = parameters.get("use_batch_fc")
    use_dropout_fc = parameters.get("use_dropout_fc")
    dropout_rate_fc = parameters.get("dropout_rate_fc")

    #########################################################################################################
    # FGE parameters
    #########################################################################################################
    # FGE Parameters
    fge_start_epoch: int = 70
    fge_num_models: int = 13
    fge_epochs_per_cycle: int = 25
    fge_cycle_start_lr: float = 0.01
    fge_cycle_end_lr: float = fge_cycle_start_lr / 10
    
    #########################################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Generators and loaders
    #########################################################################################################

    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=augmentation_prob,
                                                **other_parameters)
    else:
        train_augmentations = []

    train_gen = CauDataGenerator(subjects=train_subjects, dataset=dataset, device=device, split="train",
                                 use_age=use_age, augmentations=train_augmentations,
                                 age_noise_prob=age_noise_prob, age_noise_level=age_noise_level)
    val_gen = CauDataGenerator(subjects=val_subjects, dataset=dataset, device=device, split="val", use_age=use_age)
    test_gen = CauDataGenerator(subjects=test_subjects, dataset=dataset, device=device, split="test", use_age=use_age)

    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_gen, batch_size=batch_size, shuffle=False)

    #########################################################################################################
    # Run experiment
    #########################################################################################################

    with mlflow.start_run(run_name=folder_name):
        # Setup MLFLOW experiment
        num_runs = 1

        for run_id in range(num_runs):
            mlflow.start_run(run_name=f"{experiment}_run_{str(run_id)}", nested=True)
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

            classifier = FGEClassifier(model_name=model_name, **hyperparameters)
            train_history, val_history = get_history_objects(train_loader=train_loader, val_loader=val_loader,
                                                             save_path=save_path, num_classes=dataset.num_classes)
            try:
                model_weight_list = classifier.fit_model(train_loader=train_loader, training_epochs=train_epochs,
                                                         device=device,
                                                         loss_fn=criterion, earlystopping_patience=earlystopping,
                                                         val_loader=val_loader, train_hist=train_history,
                                                         val_history=val_history,
                                                         fge_start_epoch=fge_start_epoch, fge_num_models=fge_num_models,
                                                         fge_epochs_per_cycle=fge_epochs_per_cycle,
                                                         fge_cycle_start_lr=fge_cycle_start_lr,
                                                         fge_cycle_end_lr=fge_cycle_end_lr)
            except torch.cuda.OutOfMemoryError as e:
                mlflow.set_tag("Exception", "CUDA Out of Memory Error")
                mlflow.log_param("Exception Message", str(e))
                cleanup_function(experiment_path=experiment_path)
                print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
                break
            else:

                train_history.save_to_mlflow()
                train_history.save_to_pickle()
                val_history.save_to_mlflow()
                val_history.save_to_pickle()

                classifiers = []

                # Load all the models from the weigh lists
                for m_weights in model_weight_list:
                    classifer = FGEClassifier(model_name=model_name, pretrained=m_weights, **hyperparameters)
                    classifiers.append(classifer.to(device))

                # For each classifier, test the model and save the history
                for i, cl in enumerate(classifiers):
                    print(f"\nTesting classifier {i + 1} of {len(classifiers)}. ")
                    evaluation_history_val = History(num_classes=dataset.num_classes, set_name=f"test_val_{i}",
                                                     loader_lenght=len(val_loader), save_path=run_path)
                    classifier.test_model(test_loader=val_loader, device=device, test_hist=evaluation_history_val,
                                          loss_fn=criterion)

                    evaluation_history_test = History(num_classes=dataset.num_classes, set_name=f"test_{i}",
                                                      loader_lenght=len(test_loader), save_path=run_path)
                    classifier.test_model(test_loader=test_loader, device=device, test_hist=evaluation_history_test,
                                          loss_fn=criterion)

                    evaluation_history_val.save_to_mlflow()
                    evaluation_history_val.save_to_pickle()
                    evaluation_history_test.save_to_mlflow()
                    evaluation_history_test.save_to_pickle()

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
