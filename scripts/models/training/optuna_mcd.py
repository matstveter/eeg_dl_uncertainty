import argparse
import os
import pickle
import random

import mlflow
import numpy
import optuna
import torch
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History, get_history_objects
from eegDlUncertainty.experiments.utils_exp import cleanup_function, create_run_folder, get_parameters_from_config, \
    prepare_experiment_environment, \
    setup_experiment_path
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier

study_name = "hyper_search_test"
optuna_dir = "/home/tvetern/PhD/dl_uncertainty/optuna_dir/"
pickle_file_path = os.path.join(optuna_dir, f"{study_name}.pkl")
save_frequency = 5


def save_study(study, trial):
    if trial.number % save_frequency == 0:
        with open(pickle_file_path, "wb") as f:
            pickle.dump(study, f)
        print(f"\n\nStudy saved at trial number: {trial.number}\n\n")


def objective(trial, fixed_params):
    print("Running experiment number: ", trial.number)
    #########################################################################################################
    # Fixed parameters
    #########################################################################################################
    config_path = fixed_params.get("config_path")
    save_path = fixed_params.get("save_path")
    model_name = fixed_params.get("model_name")
    metric = fixed_params.get("metric")
    prediction = fixed_params.get("prediction")
    dataset_version = fixed_params.get("dataset_version")
    eeg_epochs = fixed_params.get("eeg_epochs")
    overlapping_epochs = fixed_params.get("overlapping_epochs")
    num_seconds = fixed_params.get("num_seconds")
    use_age = fixed_params.get("use_age")
    age_scaling = fixed_params.get("age_scaling")
    learning_rate = fixed_params.get("learning_rate")
    batch_size = fixed_params.get("batch_size")
    # augmentations = fixed_params.get("augmentations")
    # augmentation_prob = fixed_params.get("augmentation_prob")
    train_epochs = fixed_params.get("train_epochs")
    earlystopping = fixed_params.get("earlystopping")
    depth = fixed_params.get("depth")
    cnn_units = fixed_params.get("cnn_units")
    max_kernel_size = fixed_params.get("max_kernel_size")
    # num_fc_layers = fixed_params.get("num_fc_layers")
    # neurons_fc = fixed_params.get("neurons_fc")
    # use_batch_fc = fixed_params.get("use_batch_fc")
    # use_dropout_fc = fixed_params.get("use_dropout_fc")
    # dropout_rate_fc = fixed_params.get("dropout_rate_fc")
    #########################################################################################################
    #########################################################################################################
    # Other parameters
    #########################################################################################################
    age_noise_prob = 0.75
    age_noise_level = 0.05
    augmentations = ['timereverse', 'smoothtimemask']
    other_parameters = {'smoothtimemask': {'mask_len_samples': 20}}
    augmentation_prob = 0.5
    #########################################################################################################

    #########################################################################################################
    # Random states
    #########################################################################################################
    # Define random state and device
    random_state: int = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(random_state)
    numpy.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #########################################################################################################

    #########################################################################################################
    # Optuna
    #########################################################################################################
    mc_dropout_enabled = True
    num_fc_layers = trial.suggest_categorical("num_fc_layers", [1, 2, 3, 4])
    neurons_fc = trial.suggest_categorical("neurons_fc", [8, 16, 32, 64, 128, 256])
    use_batch_fc = trial.suggest_categorical("use_batch_fc", [True, False])
    use_dropout_fc = True
    dropout_rate_fc = round(trial.suggest_float("dropout_rate_fc", 0.1, 0.5, step=0.1), 1)
    mc_dropout_rate = round(trial.suggest_float("mc_dropout_rate", 0.1, 0.9, step=0.1), 1)
    #########################################################################################################

    #########################################################################################################
    # MLFlow
    #########################################################################################################
    params = {
        "num_fc_layers": num_fc_layers,
        "neurons_fc": neurons_fc,
        "use_batch_fc": use_batch_fc,
        "use_dropout_fc": use_dropout_fc,
        "dropout_rate_fc": dropout_rate_fc,
        "mc_dropout_enabled": mc_dropout_enabled,
        "mc_dropout_rate": mc_dropout_rate
    }
    print("Parameters: ", params)

    # Start mlflow run and create a folder for the current run
    mlflow.start_run(run_name=f"run_{str(trial.number)}", nested=True)
    run_path = create_run_folder(path=save_path, index=str(trial.number))

    mlflow.log_params(params)
    #########################################################################################################

    #########################################################################################################
    # Dataset
    #########################################################################################################
    dataset = CauEEGDataset(dataset_version=dataset_version, targets=prediction, eeg_len_seconds=num_seconds,
                            epochs=eeg_epochs, overlapping_epochs=overlapping_epochs, age_scaling=age_scaling,
                            save_dir=run_path)
    train_subjects, val_subjects, _ = dataset.get_splits()
    if "test" in config_path:
        train_subjects = train_subjects[0:10]
        val_subjects = val_subjects[0:10]

    weight_tensor = dataset.get_class_weights(subjects=train_subjects, normalize=True)
    weight_tensor = weight_tensor.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    #########################################################################################################
    
    #########################################################################################################
    # Generators
    #########################################################################################################
    if augmentations:
        train_augmentations = get_augmentations(aug_names=augmentations, probability=augmentation_prob,
                                                random_state=random_state, **other_parameters)
    else:
        train_augmentations = []

    train_gen = CauDataGenerator(subjects=train_subjects, dataset=dataset, device=device, split="train",
                                 use_age=use_age, augmentations=train_augmentations,
                                 age_noise_prob=age_noise_prob, age_noise_level=age_noise_level)
    val_gen = CauDataGenerator(subjects=val_subjects, dataset=dataset, device=device, split="val",
                               use_age=use_age)
    
    #########################################################################################################
    # Loaders
    #########################################################################################################
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)

    #########################################################################################################
    # Model hyperparameters
    #########################################################################################################
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
                       "mc_dropout_enabled": mc_dropout_enabled,
                       "mc_dropout_rate": mc_dropout_rate
                       }
    #########################################################################################################

    #########################################################################################################
    # Training
    #########################################################################################################
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
        raise optuna.exceptions.TrialPruned()
    except RuntimeError as e:
        if "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
            # Handle CUBLAS_STATUS_ALLOC_FAILED specifically
            torch.cuda.empty_cache()
            mlflow.set_tag("Exception", "CUBLAS Alloc Failed")
            mlflow.log_param("Exception Message", str(e))
            cleanup_function(experiment_path=run_path)
            print(f"CUBLAS Alloc Failed -> Cleanup -> Error message: {e}")
        raise optuna.exceptions.TrialPruned()
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
    # Optuna specific variables
    #########################################################################################################
    if args.config_path == "test_conf.json":
        experiment_name = "test2"
    else:
        experiment_name = f"optuna_mcd"

    global study_name
    global pickle_file_path

    study_name = f"{experiment_name}.db"
    pickle_file_path = os.path.join(optuna_dir, f"{study_name}.pkl")
    stud_path = os.path.join(optuna_dir, f"{study_name}")

    #########################################################################################################
    # Init variables from config file
    #########################################################################################################
    use_test_set: bool = parameters.pop("use_test_set", False)
    save_path: str = parameters.pop("save_path")
    model_name: str = parameters.get("classifier_name")

    metric: str = "mcc"

    if metric not in ("loss", "accuracy", "auc", "mcc"):
        raise ValueError("metric must be either 'loss', 'accuracy', 'auc', 'mcc'!")

    if metric == "loss":
        direction = "minimize"
    else:
        direction = "maximize"

    experiment_path, folder_name = setup_experiment_path(save_path=save_path,
                                                         config_path=config_path,
                                                         experiment=experiment_name)

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
        "num_seconds": parameters.get("num_seconds"),
        "use_age": parameters.get("use_age"),
        "age_scaling": parameters.get("age_scaling"),
        "learning_rate": parameters.get("learning_rate"),
        "batch_size": parameters.get("batch_size"),
        "augmentations": parameters.get("augmentations"),
        "augmentation_prob": parameters.get("augmentation_prob"),
        "train_epochs": parameters.get("training_epochs"),
        "earlystopping": parameters.get("earlystopping"),
        "depth": parameters.get("depth"),
        "cnn_units": parameters.get("cnn_units"),
        "max_kernel_size": parameters.get("max_kernel_size"),
        "num_fc_layers": parameters.get("num_fc_layers"),
        "neurons_fc": parameters.get("neurons_fc"),
        "use_batch_fc": parameters.get("use_batch_fc"),
        "use_dropout_fc": parameters.get("use_dropout_fc"),
        "dropout_rate_fc": parameters.get("dropout_rate_fc"),
        "optuna_experiment_name": parameters.get("optuna_experiment")
    }
    #########################################################################################################
    # Prepare the environment
    #########################################################################################################
    experiment_name = f"{experiment_name}"
    prepare_experiment_environment(experiment_name=experiment_name)
    number_of_trials = 175
    #########################################################################################################

    #########################################################################################################
    # Run the optimization
    #########################################################################################################
    with mlflow.start_run(run_name=experiment_name):
        # Create a study if it does not exist
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, "rb") as f:
                study = pickle.load(f)
            print(f"Study loaded from {pickle_file_path}")
        else:
            study = optuna.create_study(study_name=experiment_name, storage=f"sqlite:///{stud_path}",
                                        direction=fixed_params["direction"], load_if_exists=True)
        completed_trials = 0
        while completed_trials < number_of_trials:
            study.optimize(lambda trial: objective(trial, fixed_params),
                           n_trials=1,
                           callbacks=[save_study])
            completed_trials = sum([t.state == optuna.trial.TrialState.COMPLETE for t in study.trials])
            print("Completed trials: ", completed_trials)

        # Save the study
        with open(pickle_file_path, "wb") as f:
            pickle.dump(study, f)

        # Log the best parameters
        for k, v in study.best_params.items():
            mlflow.log_param(k, v)


if __name__ == "__main__":
    main()
