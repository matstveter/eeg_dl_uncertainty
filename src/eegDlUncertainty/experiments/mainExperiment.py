import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
import random
from typing import List, Optional, Tuple, Union

import numpy

import torch
import mlflow
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History, MCHistory
from eegDlUncertainty.data.results.plotter import Plotter
from eegDlUncertainty.data.results.utils_mlflow import add_config_information, get_experiment_name
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier
from eegDlUncertainty.models.classifiers.swag_classifier import SWAClassifier, SWAGClassifier


class BaseExperiment(ABC):
    def __init__(self, **kwargs):
        print("-----------------------------")
        self.param = kwargs.copy()
        # Get paths
        self.config_path: str = kwargs.pop("config_path")
        self.save_path: str = kwargs.pop("save_path", "/home/tvetern/PhD/dl_uncertainty/results")

        # Get names
        self.run_name: Optional[str] = kwargs.pop("run_name", None)
        self.model_name: str = kwargs.get("classifier_name")
        self.experiment_name: Optional[str] = kwargs.pop("experiment_name", None)

        # Get prediction related parameters
        self.prediction_type: str = kwargs.pop("prediction_type")
        self.pairwise: Tuple[str] = kwargs.pop("pairwise_class")
        self.one_class: str = kwargs.pop("which_one_vs_all")
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
        self.swa_start: int = kwargs.pop('swa_start')
        self.swa_lr: float = kwargs.pop('swa_lr')
        self.swa_freq: int = kwargs.pop('swa_freq')

        self.swag_enabled: bool = kwargs.pop("swag_enabled")

        self.kwargs = kwargs.copy()

        # Set values and prepare for experiments
        self.random_state: int = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths: str = self.setup_experiment_paths()
        self.prepare_experiment_environment()

        # Create values that are being initialized somewhere in the class
        self.dataset = None
        self.criterion = None
        self.train_history = None
        self.val_history = None
        self.test_history = None
        self.best_model_test_history = None
        self.mc_history = None
        self.model = None
        self.temperature_model = None
        self.swa_g_classifier = None

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

    def prepare_experiment_environment(self):
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
        torch.manual_seed(self.random_state)
        get_experiment_name(prediction_type=self.prediction_type, pairwise=self.pairwise, one_class=self.one_class,
                            experiment_name=self.experiment_name)

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
                                     prediction_type=self.prediction_type,
                                     which_one_vs_all=self.one_class,
                                     pairwise=self.pairwise,
                                     age_scaling=self.age_scaling)
        train_subjects, val_subjects, test_subjects = self.dataset.get_splits()

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

        self.__create_history(len_train=len(train_loader),
                              len_val=len(val_loader),
                              len_test=len(test_loader))

        if self.dataset.num_classes == 1:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        return train_loader, val_loader, test_loader

    def __create_history(self, len_train: int, len_val: int, len_test: int) -> None:
        """
        Initializes history objects for training, validation, test, and best model test datasets.

        This method creates four `History` instances corresponding to the training, validation,
        test, and best model test datasets. Each `History` instance is initialized with the number of
        classes in the dataset, a set name identifier, the length of the respective data loader, and
        a path for saving the history. The `History` instances are stored as attributes of the class.

        Parameters
        ----------
        len_train : int
            Length of the training data loader, indicating how many batches of data it contains.
        len_val : int
            Length of the validation data loader, indicating how many batches of data it contains.
        len_test : int
            Length of the test data loader, indicating how many batches of data it contains.

        Notes
        -----
        The `History` class is assumed to require the number of classes in the dataset, a set name,
        the loader length, and a save path as its initialization arguments. The `dataset` and `paths`
        attributes should exist within the same class as this method, storing dataset information and
        history saving path respectively.
        """
        if self.dataset is not None:
            self.train_history = History(num_classes=self.dataset.num_classes, set_name="train",
                                         loader_lenght=len_train,
                                         save_path=self.paths)
            self.val_history = History(num_classes=self.dataset.num_classes, set_name="val",
                                       loader_lenght=len_val,
                                       save_path=self.paths)
            self.test_history = History(num_classes=self.dataset.num_classes, set_name="test", loader_lenght=len_test,
                                        save_path=self.paths)
            self.best_model_test_history = History(num_classes=self.dataset.num_classes, set_name="best_test",
                                                   loader_lenght=len_test,
                                                   save_path=self.paths)
            if self.mc_dropout_enabled:
                self.mc_history = MCHistory(save_path=self.paths, num_classes=self.dataset.num_classes)
        else:
            raise ValueError("Dataset not provided!")

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @staticmethod
    def get_model(model_name, pretrained=None, **kwargs):
        return MainClassifier(model_name=model_name, pretrained=pretrained, **kwargs)

    def train(self, train_loader, val_loader):
        """
        Trains the model using the provided training and validation data loaders.

        This method starts the training process of the model by calling its `fit_model` method with the
        training and validation data loaders, the number of training epochs, the computing device, the loss function,
        and history objects for both training and validation. It effectively encapsulates the training loop,
        directing data through the model and making adjustments based on the computed loss and the specified
        optimization strategy.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader for the training dataset, providing batches of data.
        val_loader : DataLoader
            The DataLoader for the validation dataset, used for evaluating the model's performance
            on unseen data during training.

        Notes
        -----
        - The model is expected to have a `fit_model` method with parameters for the training and validation
          data loaders, number of epochs, device, loss function, and history objects for training and validation.
        - This method assumes that `self.train_epochs`, `self.device`, `self.criterion`, `self.train_history`, and
          `self.val_history` are already properly initialized in the class.
        - The actual training logic, including the forward pass, loss computation, backpropagation, and parameter
          updates, is handled within the `fit_model` method of the model. This method primarily serves to organize
          these operations and pass the necessary configurations.
        - The method does not return a value but is expected to update the model's weights and the history objects
          with the results from each epoch of training and validation.
        """
        if self.model is not None:
            self.model.fit_model(train_loader=train_loader, val_loader=val_loader,
                                 training_epochs=self.train_epochs,
                                 device=self.device,
                                 loss_fn=self.criterion,
                                 train_hist=self.train_history,
                                 val_history=self.val_history,
                                 earlystopping_patience=self.earlystopping)
        else:
            raise ValueError("Model is not initialized and is None!")

    def test_models(self, test_loader, use_temp_scaling=False, val_loader=None):
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

        if (self.model is not None and self.criterion is not None and self.device is not None and self.test_history is
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
            best_model = self.get_model(model_name=self.model_name,
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

    def temperature_scaling(self, val_loader):
        if self.model is not None:
            self.model.set_temperature(val_loader=val_loader, criterion=self.criterion, device=self.device)
        else:
            raise ValueError("Model is None!")

    def mc_dropout(self, test_loader):
        if self.mc_history is None:
            raise ValueError("MC History object not initialized!")

        self.model.get_mc_predictions(test_loader=test_loader, device=self.device, history=self.mc_history)

    def swa_g_train(self, train_loader, val_loader):

        if self.swa_enabled:
            self.swa_g_classifier = SWAClassifier(pretrained_model=self.model, learning_rate=self.learning_rate,
                                                  save_path=self.paths,
                                                  model_hyperparameters=self.model.hyperparameters,
                                                  name=self.model_name)
            # todo Can probably use the forward method using the self.swa_g_classifier for testing the performance
            # todo Log performance using history objects...
        else:
            # todo Implement swag
            # todo Use a similar setup as SWA, can perhaps just inherit, changing the fit method only..
            self.swa_g_classifier = SWAGClassifier()

        self.swa_g_classifier.fit(train_loader=train_loader, val_loader=val_loader, swa_epochs=2,
                                  device=self.device, loss_fn=self.criterion, swa_lr=self.swa_lr)

    def conformal_prediction(self, val_loader, test_loader, conformal_algorithm, use_temp_scaling=False):
        coverage = self.model.conformal_prediction(val_loader=val_loader, test_loader=test_loader, device=self.device,
                                                   use_temp_scaling=use_temp_scaling, criterion=self.criterion,
                                                   conformal_algorithm=conformal_algorithm)
        if use_temp_scaling:
            mlflow.log_param(f"{conformal_algorithm}_coverage_with_temp_scale", coverage)
        else:
            mlflow.log_param(f"{conformal_algorithm}_Coverage_without_temp_scale", coverage)

    def finish_run(self):
        # Save the data
        if (self.train_history is not None and self.val_history is not None and
                self.test_history is not None and self.best_model_test_history is not None):
            self.train_history.save_to_pickle()
            self.val_history.save_to_pickle()
            self.test_history.save_to_pickle()

            self.train_history.save_to_mlflow()
            self.val_history.save_to_mlflow()
            self.test_history.save_to_mlflow()
            self.best_model_test_history.save_to_mlflow()

            plot = Plotter(train_dict=self.train_history.get_as_dict(),
                           val_dict=self.val_history.get_as_dict(),
                           test_dict=self.test_history.get_as_dict(),
                           test_dict_best_model=self.best_model_test_history.get_as_dict(), save_path=self.paths)
            plot.produce_plots()
        else:
            raise ValueError("History object is None, initialize train, val, test and best test history objects!")

        if self.mc_history is not None:
            self.mc_history.save_to_pickle()

    def run(self):
        """
        Executes the machine learning experiment from start to finish.

        This method encompasses the full lifecycle of a machine learning experiment, including initializing an
        MLflow run, preparing data, creating the model, training, handling out-of-memory errors, testing, and
        finalizing the experiment. It leverages MLflow for experiment tracking, allowing for the recording of
        parameters, metrics, and artifacts for analysis and reproducibility.

        The process includes:
        - Starting an MLflow run with a specific name, if provided.
        - Preparing data loaders for training, validation, and testing datasets.
        - Creating the model with specified parameters.
        - Training the model and handling any CUDA out-of-memory errors by logging the exception in MLflow and
          performing cleanup operations.
        - If training succeeds without errors, testing the model on a test dataset.
        - Finalizing the experiment by completing the MLflow run.

        Notes
        -----
        - The method assumes that `self.run_name`, `self.prepare_data`, `self.create_model`, `self.train`,
          `self.test_models`, and `self.finish_run` are properly defined and implemented.
        - Out-of-memory errors during training are specifically caught and handled. Such errors trigger cleanup
          operations and logging of the error details to MLflow before the script halts or proceeds.
        - MLflow is used to track the experiment's details, including starting and ending runs, tagging exceptions,
          and logging parameters or metrics. This requires MLflow to be correctly set up and configured in the
          environment.
        - The method automatically ends the MLflow run in a `finally` block, ensuring that the run is closed
          properly even if errors occur.

        Examples
        --------
        >>> self.run()
        Initiates the experiment, automatically managing the MLflow run, data preparation, model training/testing,
        and error handling.
        """
        if self.run_name is not None:
            mlflow.start_run(run_name=self.run_name)
        else:
            mlflow.start_run()
        add_config_information(config=self.param, dataset="CAUEEG")

        train_loader, val_loader, test_loader = self.prepare_data()
        self.create_model(**self.kwargs)
        try:
            self.train(train_loader=train_loader, val_loader=val_loader)
        except torch.cuda.OutOfMemoryError as e:
            mlflow.set_tag("Exception", "CUDA Out of Memory Error")
            mlflow.log_param("Exception Message", str(e))
            self.cleanup_function()
            print(f"Cuda Out Of Memory -> Cleanup -> Error message: {e}")
        else:
            if self.swag_enabled or self.swag_enabled:
                self.swa_g_train(train_loader=train_loader, val_loader=val_loader)

            self.test_models(test_loader=test_loader, use_temp_scaling=False, val_loader=val_loader)

            if self.mc_dropout_enabled and not self.swag_enabled:
                self.mc_dropout(test_loader=test_loader)

            if not self.swag_enabled:
                self.conformal_prediction(val_loader=val_loader, test_loader=test_loader, use_temp_scaling=True,
                                          conformal_algorithm="APS")
            # self.conformal_prediction(val_loader=val_loader, test_loader=test_loader, use_temp_scaling=True,
            #                           conformal_algorithm="RAPS")
            # self.conformal_prediction(val_loader=val_loader, test_loader=test_loader, use_temp_scaling=True,
            #                           conformal_algorithm="equal_weighted")

            self.finish_run()

        finally:
            mlflow.end_run()

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
