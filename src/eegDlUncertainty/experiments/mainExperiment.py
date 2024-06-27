import os
import random
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

import mlflow
import numpy
import torch
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.utils_mlflow import add_config_information, get_experiment_name
from eegDlUncertainty.models.classifiers.swag_classifier import SWAClassifier


class BaseExperiment(ABC):
    def __init__(self, **kwargs):
        print("-----------------------------")
        self.param = kwargs.copy()

        self.use_test_set = kwargs.pop("use_test_set")

        # Get paths
        self.config_path: str = kwargs.pop("config_path")
        self.save_path: str = kwargs.pop("save_path", "/home/tvetern/PhD/dl_uncertainty/results")

        # Get names
        self.run_name: Optional[str] = kwargs.pop("run_name", None)
        self.model_name: str = kwargs.get("classifier_name")
        self.experiment_name: Optional[str] = kwargs.pop("experiment_name", None)

        # Get prediction related parameters
        self.dataset_version: int = kwargs.pop("dataset_version")
        self.prediction: str = kwargs.pop("prediction")
        self.use_age: bool = kwargs.pop('use_age')
        self.age_scaling: str = kwargs.pop('age_scaling')

        # Get eeg related parameters
        self.num_seconds: int = kwargs.pop("num_seconds")
        self.eeg_epochs: int = kwargs.pop("eeg_epochs")
        self.overlapping_epochs: bool = kwargs.pop('epoch_overlap')

        # Get augmentation related parameters
        self.augmentations: List[Union[str, None]] = kwargs.pop("augmentations")
        self.augmentation_prob: Optional[float] = kwargs.pop("augmentation_prob", 0.2)

        # Set training specific data
        self.train_epochs: int = kwargs.get("training_epochs")
        self.batch_size: int = kwargs.get("batch_size")
        self.learning_rate: float = kwargs.get("learning_rate")
        self.earlystopping: int = kwargs.get("earlystopping")

        self.mc_dropout_enabled: bool = kwargs.get("mc_dropout_enabled")

        # SWA specific variables
        self.swa_enabled: bool = kwargs.pop('swa_enabled')
        self.swa_lr: float = kwargs.pop('swa_lr')
        self.swa_epochs: int = kwargs.pop('swa_epochs')

        self.swag_enabled: bool = kwargs.pop("swag_enabled")
        self.swag_lr: float = kwargs.pop("swag_lr")
        self.swag_freq: int = kwargs.pop("swag_freq")

        self.kwargs = kwargs.copy()

        # Set values and prepare for experiments
        self.random_state: int = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths: str = self.setup_experiment_paths()
        self.prepare_experiment_environment()

        # Create values that are being initialized somewhere in the class
        self.dataset = None
        self.criterion = None
        self.model = None
        self.temperature_model = None
        self.test_subjects = None

    def prepare_run(self):
        if self.run_name is not None:
            mlflow.start_run(run_name=self.run_name)
        else:
            mlflow.start_run()
        add_config_information(config=self.param, dataset="CAUEEG")
        train_loader, val_loader, test_loader = self.prepare_data()
        self.add_dataset_hyperparameters()
        return train_loader, val_loader, test_loader

    def setup_experiment_paths(self):
        """
        Sets up and returns the directory path for saving experiment files.

        This method generates a unique directory path by appending the model name and the current timestamp to the base
        save path specified by `self.save_path`. It then creates this directory (if it doesn't already exist) and copies
        the configuration file specified by `self.config_path` into it. The configuration file is assumed to reside in a
        `config_files` directory located in the same directory as this script.

        Returns
        -------
        str
            The path to the newly created directory where experiment files will be saved.

        Notes
        -----
        - The method uses the `os` module to manipulate paths and create directories, and `shutil` for file operations.
        - It is assumed that `self.save_path`, `self.model_name`, and `self.config_path` are already set and point to
        valid locations or files.
        - The unique directory name is generated using the model name and the current timestamp, which provides a simple
          way to keep experiment outputs organized and avoid overwriting previous results.
        - This method does not handle errors that might occur during directory creation or file copying, so additional
          error handling may be necessary depending on the application's requirements.
        """
        paths = os.path.join(self.save_path, f"{self.model_name}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}")
        os.makedirs(paths, exist_ok=True)
        shutil.copy(src=os.path.join(os.path.dirname(__file__), "config_files", self.config_path),
                    dst=os.path.join(paths, os.path.basename(self.config_path)))
        return paths

    def prepare_experiment_environment(self, seed=None):
        """
        Prepares the environment for running a machine learning experiment.

        This method configures the MLflow tracking URI for experiment logging, enables system metrics logging via an
        environment variable, sets the random seeds for Python's built-in `random` module and NumPy to ensure
        reproducibility, and determines the experiment name based on prediction characteristics.

        The MLflow tracking URI is set to a predefined path, and system metrics logging is enabled to record additional
        system-level information during the experiment. Random seeds are synchronized with the instance's `random_state`
        attribute, affecting random number generation in both Python and NumPy, thereby ensuring that the experiment's
        stochastic elements are reproducible.

        Finally, it calls `get_experiment_name` with parameters that specify the type of prediction, whether the
        prediction is pairwise, and if it is a one-class prediction, to dynamically generate and presumably set
        an experiment name based on these characteristics.

        Notes
        -----
        - The method assumes that `mlflow`, `os`, `random`, and `numpy` modules are imported.
        - The `random_state`, `prediction_type`, `pairwise`, and `one_class` attributes must be set before this method
        is called.
        - It's implied that `get_experiment_name` is a function that sets or uses the experiment name in some capacity,
          though the mechanism for this is not specified within this method.
        """
        mlflow.set_tracking_uri("file:///home/tvetern/PhD/dl_uncertainty/results/mlflow/")
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        random.seed(self.random_state)
        numpy.random.seed(self.random_state)
        if seed is None:
            torch.manual_seed(self.random_state)
        else:
            torch.manual_seed(seed)
        get_experiment_name(experiment_name=self.experiment_name)

    def prepare_data(self):
        """
        Prepares the data for training, validation, and testing, and sets the loss function.

        This method initializes the dataset with specified parameters, splits it into training, validation,
        and testing subsets, and sets up corresponding data loaders. If specified, data augmentation is applied to
        the training data. It also initializes history objects for recording training, validation, and testing
        performance, and selects an appropriate loss function based on the number of classes in the dataset.

        The method concludes by returning data loaders for the training, validation, and testing sets.

        Returns
        -------
        tuple
            A tuple containing data loaders for the training, validation, and testing datasets respectively.

        Notes
        -----
        - This method assumes that the dataset class `CauEEGDataset` and the data generator class `CauDataGenerator`
          are available and correctly implement the required functionality for dataset manipulation and loading.
        - Data augmentation is optional and applied to the training data if `self.augmentations` is not empty. The
          augmentations are specified by name and configured with a given probability and a random state.
        - History objects for training, validation, and testing are initialized through a call to
          `self.__create_history`, using the lengths of the corresponding data loaders to set up these histories.
        - The loss function is chosen based on the number of classes in the dataset: `BCEWithLogitsLoss` for binary
          classification (one class) and `CrossEntropyLoss` for multi-class classification.
        - The device for running the dataset generation and data loaders is specified by `self.device`.
        - The method handles setting up data loaders with and without augmentations, and ensures shuffling of the data.

        Examples
        --------
        >>> t_loader, v_loader, te_loader = self.prepare_data()
        This will prepare the data loaders and return them for use in training, validation, and testing phases.
        """
        self.dataset = CauEEGDataset(dataset_version=self.dataset_version,
                                     targets=self.prediction,
                                     eeg_len_seconds=self.num_seconds,
                                     epochs=self.eeg_epochs,
                                     overlapping_epochs=self.overlapping_epochs,
                                     age_scaling=self.age_scaling)
        train_subjects, val_subjects, test_subjects = self.dataset.get_splits()

        self.test_subjects = test_subjects

        # Set up the training data generator and loader
        train_gen = CauDataGenerator(subjects=train_subjects, dataset=self.dataset, device=self.device, split="train",
                                     use_age=self.use_age)
        if self.augmentations:
            train_augmentations = get_augmentations(aug_names=self.augmentations, probability=self.augmentation_prob,
                                                    random_state=self.random_state)
            # noinspection PyTypeChecker
            train_loader = AugmentedDataLoader(dataset=train_gen, transforms=train_augmentations, device=self.device,
                                               batch_size=self.batch_size,
                                               shuffle=True)
        else:
            train_loader = DataLoader(train_gen, batch_size=self.batch_size, shuffle=True)

        # Set up the validation data generator and loader
        val_gen = CauDataGenerator(subjects=val_subjects, dataset=self.dataset, device=self.device, split="val",
                                   use_age=self.use_age)
        val_loader = DataLoader(val_gen, batch_size=self.batch_size, shuffle=True)

        # Test data generator and loader
        test_gen = CauDataGenerator(subjects=test_subjects, dataset=self.dataset, device=self.device, split="test",
                                    use_age=self.use_age)
        test_loader = DataLoader(test_gen, batch_size=self.batch_size, shuffle=False)

        if self.dataset.num_classes == 1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        return train_loader, val_loader, test_loader

    def _get_history(self, train_loader, val_loader, test_loader):
        train_history = get_history_object(num_classes=self.dataset.num_classes, loader_length=len(train_loader),
                                           set_name="train", save_path=self.paths)
        val_history = get_history_object(num_classes=self.dataset.num_classes, loader_length=len(val_loader),
                                         set_name="val", save_path=self.paths)
        test_history = get_history_object(num_classes=self.dataset.num_classes, loader_length=len(test_loader),
                                          set_name="test", save_path=self.paths)
        return train_history, val_history, test_history

    def add_dataset_hyperparameters(self):
        if self.dataset is not None:
            hyperparameters = {"in_channels": self.dataset.num_channels,
                               "num_classes": self.dataset.num_classes,
                               "time_steps": self.dataset.eeg_len,
                               "save_path": self.paths,
                               "lr": self.learning_rate}
            self.kwargs.update(hyperparameters)
        else:
            raise ValueError("Dataset is not provided!")

    def cleanup_function(self):
        """
        Attempts to delete the specified folder and its contents.

        Notes
        -----
        - This function uses `shutil.rmtree` to delete the folder and all its contents.
        - Error handling is implemented to catch and log exceptions, preventing the function from raising exceptions if
          the folder cannot be deleted (e.g., if the folder does not exist or an error occurs during deletion).
        """
        try:
            shutil.rmtree(self.paths)
            print(f"Successfully deleted the folder: {self.paths} due to OOMemory")
        except Exception as e:
            print(f"Error deleting the folder: {self.paths}. Exception: {e}")

    def temperature_scaling(self, val_loader):
        if self.model is not None:
            self.model.set_temperature(val_loader=val_loader, criterion=self.criterion, device=self.device)
        else:
            raise ValueError("Model is None!")

    def swa_train(self, train_loader, val_loader, history):

        swa_classifier = SWAClassifier(pretrained_model=self.model, learning_rate=self.learning_rate,
                                       save_path=self.paths,
                                       model_hyperparameters=self.model.hyperparameters,
                                       name=self.model_name)
        swa_classifier.fit(train_loader=train_loader, val_loader=val_loader, swa_epochs=self.swa_epochs,
                           device=self.device, loss_fn=self.criterion, swa_lr=self.swa_lr)
        return swa_classifier

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @abstractmethod
    def train(self, train_loader, val_loader, train_history, val_history):
        pass

    def test_models(self, test_loader, val_loader, use_temp_scaling=False):
        """
        Evaluates the current and the best performing models on the test dataset.

        This method conducts a two-stage testing process. Initially, it evaluates the current model's performance
        using the provided test data loader, saving the results in the specified test history object. Subsequently,
        it loads the best performing model based on a predefined criterion from storage and repeats the evaluation
        process, storing its performance in a separate history object designated for the best model's test results.

        Parameters
        ----------
        test_loader : DataLoader
            The DataLoader providing the test dataset, used for evaluating the models.

        Notes
        -----
        - The method assumes the presence of a `test_model` method within the model object, capable of evaluating
          the model on the test dataset and recording the results in the provided history object.
        - `self.model` should be an instance of a model class with predefined methods for testing and loading
          pretrained weights, as well as attributes for model path and hyperparameters.
        - The method leverages `self.test_history` and `self.best_model_test_history` to record the testing
          outcomes of the current and best models, respectively.
        - It initializes the best model using the `MainClassifier` class, with the `classifier_name` set to the
          model's name, `pretrained` set to the path of the best model's weights, and additional model
          hyperparameters passed directly.
        - The evaluation metrics and the mechanism for determining the 'best model' are not specified in this method
          but are important for understanding how the best model is selected and evaluated.

        Examples
        --------
        Assuming `test_loader` is an instance of `DataLoader` with your test dataset:

        >>> test_models(test_loader)
        This will evaluate both the current and the best performing model on the test dataset, recording their
        performance metrics in their respective history objects.
        """

        if self.use_test_set:
            if (
                    self.model is not None and self.criterion is not None and self.device is not None and self.test_history is
                    not None and self.best_model_test_history is not None):
                if use_temp_scaling and val_loader is None:
                    raise ValueError(
                        "If temperature scaling is set to True, the validation data loader must also be sent to "
                        "the test_models function.")

                if use_temp_scaling:
                    self.model.set_temperature(val_loader=val_loader, criterion=self.criterion, device=self.device,
                                               model_name="Last Model")
                self.model.test_model(test_loader=test_loader, test_hist=self.test_history,
                                      device=self.device, loss_fn=self.criterion)

                # load the best model
                best_model = self.get_model(model_name=self.model_name, mc_model=self.mc_dropout_enabled,
                                            pretrained=self.model.model_path(with_ext=True),
                                            **self.model.hyperparameters)
                if use_temp_scaling:
                    best_model.set_temperature(val_loader=val_loader, criterion=self.criterion, device=self.device,
                                               model_name="Best Model")

                best_model.test_model(test_loader=test_loader,
                                      test_hist=self.best_model_test_history,
                                      device=self.device,
                                      loss_fn=self.criterion)
            else:
                raise ValueError(f"Missing initialization of values!\n"
                                 f"{self.model=}\n"
                                 f"{self.criterion=}\n"
                                 f"{self.device=}\n"
                                 f"{self.test_history=}\n"
                                 f"{self.best_model_test_history=}\n")
        else:
            print("Testing with the best model on the validation set!")
            best_model = self.get_model(model_name=self.model_name,
                                        pretrained=self.model.model_path(with_ext=True),
                                        **self.model.hyperparameters)
            best_model.test_model(test_loader=val_loader,
                                  test_hist=self.best_model_test_history,
                                  device=self.device,
                                  loss_fn=self.criterion)
