import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
import random
import numpy

import torch
import mlflow
from braindecode.augmentation import AugmentedDataLoader
from torch.utils.data import DataLoader

from eegDlUncertainty.data.data_generators.CauDataGenerator import CauDataGenerator
from eegDlUncertainty.data.data_generators.augmentations import get_augmentations
from eegDlUncertainty.data.dataset.CauEEGDataset import CauEEGDataset
from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.data.results.plotter import Plotter
from eegDlUncertainty.data.results.utils_mlflow import add_config_information, get_experiment_name
from eegDlUncertainty.models.classifiers.main_classifier import MainClassifier


class BaseExperiment(ABC):
    def __init__(self, **kwargs):
        print("-----------------------------")
        self.param = kwargs.copy()
        # Get paths
        self.config_path = kwargs.pop("config_path")
        self.save_path = kwargs.pop("save_path", "/home/tvetern/PhD/dl_uncertainty/results")

        # Get names
        self.run_name = kwargs.pop("run_name", None)
        self.model_name = kwargs.get("classifier_name")
        self.experiment_name = kwargs.pop("experiment_name", None)

        # Get prediction related parameters
        self.prediction_type = kwargs.pop("prediction_type")
        self.pairwise = kwargs.pop("pairwise_class")
        self.one_class = kwargs.pop("which_one_vs_all")
        self.dataset_version = kwargs.pop("dataset_version")
        self.prediction = kwargs.pop("prediction")

        # Get eeg related parameters
        self.num_seconds = kwargs.pop("num_seconds")
        self.eeg_epochs = kwargs.pop("eeg_epochs")

        # Get augmentation related parameters
        self.augmentations = kwargs.pop("augmentations")
        self.augmentation_prob = kwargs.pop("augmentation_prob", 0.2)

        # Set training specific data
        self.train_epochs = kwargs.get("training_epochs")
        self.batch_size = kwargs.get("batch_size")
        self.learning_rate = kwargs.get("learning_rate")

        self.kwargs = kwargs.copy()

        # Set values and prepare for experiments
        self.random_state = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = self.setup_experiment_paths()
        self.prepare_experiment_environment()

        # Create values that are being initialized somewhere in the class
        self.dataset = None
        self.criterion = None
        self.train_history = None
        self.val_history = None
        self.test_history = None
        self.best_model_test_history = None
        self.model = None

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
        >>> train_loader, val_loader, test_loader = self.prepare_data()
        This will prepare the data loaders and return them for use in training, validation, and testing phases.
        """
        self.dataset = CauEEGDataset(dataset_version=self.dataset_version,
                                     targets=self.prediction,
                                     eeg_len_seconds=self.num_seconds,
                                     epochs=self.eeg_epochs,
                                     prediction_type=self.prediction_type,
                                     which_one_vs_all=self.one_class,
                                     pairwise=self.pairwise)
        train_subjects, val_subjects, test_subjects = self.dataset.get_splits()

        # Set up the training data generator and loader
        train_gen = CauDataGenerator(subjects=train_subjects, dataset=self.dataset, device=self.device)
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
        val_gen = CauDataGenerator(subjects=val_subjects, dataset=self.dataset, device=self.device)
        val_loader = DataLoader(val_gen, batch_size=self.batch_size, shuffle=True)

        # Test data generator and loader
        test_gen = CauDataGenerator(subjects=test_subjects, dataset=self.dataset, device=self.device)
        test_loader = DataLoader(test_gen, batch_size=self.batch_size, shuffle=True)
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
        self.train_history = History(num_classes=self.dataset.num_classes, set_name="train", loader_lenght=len_train,
                                     save_path=self.paths)
        self.val_history = History(num_classes=self.dataset.num_classes, set_name="val",
                                   loader_lenght=len_val,
                                   save_path=self.paths)
        self.test_history = History(num_classes=self.dataset.num_classes, set_name="test", loader_lenght=len_test,
                                    save_path=self.paths)
        self.best_model_test_history = History(num_classes=self.dataset.num_classes, set_name="best_test",
                                               loader_lenght=len_test,
                                               save_path=self.paths)

    @abstractmethod
    def create_model(self, **kwargs):
        pass

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
        self.model.fit_model(train_loader=train_loader, val_loader=val_loader,
                             training_epochs=self.train_epochs,
                             device=self.device,
                             loss_fn=self.criterion,
                             train_hist=self.train_history,
                             val_history=self.val_history)

    def test_models(self, test_loader):
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
        self.model.test_model(test_loader=test_loader, test_hist=self.test_history,
                              device=self.device, loss_fn=self.criterion)

        # load the best model
        best_model = MainClassifier(model_name=self.model_name,
                                    pretrained=self.model.model_path(with_ext=True),
                                    **self.model.hyperparameters)
        best_model.test_model(test_loader=test_loader,
                              test_hist=self.best_model_test_history,
                              device=self.device,
                              loss_fn=self.criterion)

    def finish_run(self):
        # Save the data
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
            self.test_models(test_loader=test_loader)
            self.finish_run()

        finally:
            mlflow.end_run()

    def cleanup_function(self):
        """
        Attempts to delete the specified folder and its contents.

        Parameters
        ----------
        folder_path : str
            The path to the folder to be deleted.

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
