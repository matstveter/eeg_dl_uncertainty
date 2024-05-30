import os.path
import pickle
from typing import Any, Dict, List, Optional
import seaborn as sns
import mlflow
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (roc_auc_score, f1_score, cohen_kappa_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)

from eegDlUncertainty.experiments.utils_exp import check_folder


class History:

    def __init__(self, num_classes: int, set_name: str, loader_lenght, save_path, verbose=True):
        self._loss: List[float] = []
        self._accuracy: List[float] = []
        self._precision: List[float] = []
        self._recall: List[float] = []
        self._f1: List[float] = []
        self._auc: List[float] = []
        self._kappa: List[float] = []
        self._mcc: List[float] = []
        self._conf_mat: List[float] = []
        self._num_classes = num_classes

        self.epoch_y_true: List[torch.Tensor] = []
        self.epoch_y_pred: List[torch.Tensor] = []
        self.epoch_loss: int = 0
        self.verbose: bool = verbose
        self._set_name: str = set_name
        self._loader_lenght: int = loader_lenght
        self._save_path = save_path

    @property
    def num_classes(self) -> int:
        """ Property -> returns the number of classes in the prediction
        """
        return self._num_classes

    def get_last_loss(self) -> float:
        """ Returns the last loss
        """
        return self._loss[-1]

    def get_last_auc(self) -> float:
        """ Returns the last AUC
        """
        return self._auc[-1]

    def get_last_acc(self) -> float:
        """ Returns the last saved accuracy

        """
        return self._accuracy[-1]

    def _update_metrics(self) -> None:
        """
        Updates the performance metrics for the model using the data accumulated over an epoch.

        This method computes various metrics such as loss, precision, recall, f1 score, AUC, kappa,
        MCC, and potentially a confusion matrix if evaluating a validation set. It handles both binary
        and multi-class scenarios. Metrics are appended to their respective lists in the object state.

        For binary classification:
        - Metrics are calculated directly from the predictions and true labels.
        - AUC is calculated from probabilities.

        For multi-class:
        - Predictions are converted to one-hot format.
        - AUC is calculated considering a one-vs-rest approach.

        Metrics calculation assumes the presence of `epoch_y_pred`, `epoch_y_true`, and `epoch_loss`
        attributes populated during the epoch's processing.

        Notes
        -----
        - This method should be called at the end of each epoch to update the model's performance metrics.
        - The tensors `epoch_y_pred` and `epoch_y_true` should be tensors collected during the epoch.

        Raises
        ------
        RuntimeError
            If there are inconsistencies in tensor dimensions or incompatible operations due to
            data type errors or device mismatches.
        """
        self._loss.append(self.epoch_loss / self._loader_lenght)
        y_pred = torch.tensor(self.epoch_y_pred)
        y_true = torch.tensor(self.epoch_y_true)

        if self._num_classes == 1:
            y_pred_proba = y_pred.detach().cpu().numpy()
            y_pred = torch.round(y_pred)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            self._precision.append(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            self._recall.append(recall_score(y_true=y_true, y_pred=y_pred))
            self._f1.append(f1_score(y_true=y_true, y_pred=y_pred))
            self._auc.append(roc_auc_score(y_true=y_true, y_score=y_pred_proba))
        else:
            y_pred_proba = y_pred.detach().cpu().numpy()
            y_true_one_hot = y_true.detach().cpu().numpy()
            _, y_pred = torch.max(y_pred, dim=1)
            _, y_true = torch.max(y_true, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()
            self._precision.append(precision_score(y_true=y_true, y_pred=y_pred, average="weighted",
                                                   zero_division=0))
            self._recall.append(recall_score(y_true=y_true, y_pred=y_pred, average="weighted"))
            self._f1.append(f1_score(y_true=y_true, y_pred=y_pred, average="weighted"))
            self._auc.append(
                roc_auc_score(y_true=y_true_one_hot, y_score=y_pred_proba, multi_class="ovr", average="weighted"))

        self._kappa.append(cohen_kappa_score(y1=y_true, y2=y_pred))
        self._mcc.append(matthews_corrcoef(y_true=y_true, y_pred=y_pred))

        if self._set_name.lower() == "val":
            self._conf_mat.append(confusion_matrix(y_pred=y_pred, y_true=y_true))

        self._accuracy.append(self._calculate_accuracy(y_pred=y_pred, y_true=y_true))

    def print_metrics(self) -> None:
        """
        Prints the latest metrics for the model to the console.

        This method formats and displays the latest recorded metrics such as loss, accuracy, AUC,
        precision, recall, F1 score, and MCC. It provides a quick snapshot of model performance
        for the current data set (e.g., training, validation, test). For test sets, additional
        newlines are added before the metrics for better separation in outputs.

        Notes
        -----
        - The method assumes that all metrics are up-to-date and contain at least one recorded value
          in their respective lists.
        - Metrics are displayed in a single line; if 'test' is in `_set_name`, it prepends with newlines
          for visual clarity in grouped logging outputs.
        """
        if "test" in self._set_name:
            print("\n\n")
        print(f"{self._set_name.upper()}: Loss: {self._loss[-1]:.4f}"
              f"  Accuracy: {self._accuracy[-1]:.2f}"
              f"  AUC: {self._auc[-1]:.2f}"
              f"  Precision: {self._precision[-1]:.2f}"
              f"  Recall: {self._recall[-1]:.2f}"
              f"  F1: {self._f1[-1]:.2f}"
              f"  MCC: {self._mcc[-1]:.2f}  ", end="")

    def on_epoch_end(self) -> None:
        """ This functions updates the metrics, prints performance and prepares lists for the new epoch.

        This functions starts by calling the function self._update_metrics which updates the saved metrics with the
        performance and loss from the current epoch. If verbose is True, it will call the function print_metrics, to
        print the performance of the current epoch. Then as long as the current class is not the test set, the
        epoch list with predictions and ground truths are emptied, and the loss is set to 0 to be ready for the next
        epoch.

        Returns
        -------
        None
        """
        self._update_metrics()
        if self.verbose:
            self.print_metrics()

        if "test" not in self._set_name:
            self.epoch_y_true = []
            self.epoch_y_pred = []
            self.epoch_loss = 0

    def batch_stats(self, y_pred, y_true, loss) -> None:
        """ This function appends the batch stats to the epoch lists

        This function receives the predicted values, the ground truths and the loss as arguments. Depending on the
        prediction problem, either binary or multiple classes, it appends these values to three class variables.
        For the predicted values this is self.epoch_y_pred, for the ground truth this si self.epoch_y_true and the
        loss is added to the self.epoch_loss.

        Parameters
        ----------
        y_pred: np.ndarray
            The predicted values from the model
        y_true: np.ndarray
            The ground truth values
        loss:
            Loss calculated from the performance

        Returns
        -------
        None
        """
        # Extend epoch_y_pred and epoch_y_true based on label representation
        if self._num_classes == 1:  # Binary classification
            self.epoch_y_pred.extend(y_pred)
            self.epoch_y_true.extend(y_true)
        else:  # Multiclass classification
            self.epoch_y_pred.extend(y_pred.tolist())  # Convert predicted probabilities to list
            self.epoch_y_true.extend(y_true.tolist())  # Convert one-hot encoded

        self.epoch_loss += loss.item()

    @staticmethod
    def _calculate_accuracy(y_pred, y_true) -> float:
        """ Calculates the accuracy of the predictions

        This functions sums up the equal predictions of y_pred and the true labels and then divide by the lenght
        of y_true and lastly multiplies by 100 to make ti to percentage

        Parameters
        ----------
        y_pred: np.ndarray
            The predicted values from the model
        y_true: np.ndarray
            The true values or ground truths

        Returns
        -------
        correct_percentage: float
            Percentage of correctly classified points

        """
        correct_percentage = ((y_pred == y_true).sum() / len(y_true)) * 100
        return correct_percentage

    def get_as_dict(self) -> Dict[str, Any]:
        """
           Constructs and returns a dictionary containing various metrics and properties of a model.

           This method aggregates multiple attributes related to model performance metrics and other
           related properties into a single dictionary. It is useful for serialization or for providing
           a snapshot of model's state in a format that's easy to log or inspect.

           Returns
           -------
           Dict[str, Any]
               A dictionary containing the following key-value pairs:
               - "loss": A list or a single value representing the loss metric.
               - "accuracy": Model accuracy metric.
               - "precision": Precision metric, often used in classification tasks.
               - "recall": Recall metric, measures the ability to find all relevant instances.
               - "f1": F1 score, a harmonic mean of precision and recall.
               - "auc": Area Under the ROC Curve.
               - "kappa": Cohen’s kappa, a statistic that measures inter-annotator agreement.
               - "mcc": Matthews correlation coefficient, a measure of quality for binary classifications.
               - "conf_mat": Confusion matrix of the model's predictions.
               - "num_classes": Number of classes in the classification task.
               - "set_name": The name of the data set (e.g., 'train', 'validation', 'test').
               - "loader_length": The length of the data loader used during training or evaluation.

           Notes
           -----
           This method assumes that all metric attributes are initialized and populated properly
           prior to invocation. It is intended to be used when a complete set of metrics is available.

           Raises
           ------
           AttributeError
               If any of the expected attributes are not initialized.
           """
        return {
            "loss": self._loss,
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "auc": self._auc,
            "kappa": self._kappa,
            "mcc": self._mcc,
            "conf_mat": self._conf_mat,
            "num_classes": self._num_classes,
            "set_name": self._set_name,
            "loader_length": self._loader_lenght
        }

    def save_to_pickle(self):
        """
        Save the object's data in a dictionary format to a pickle file.

        This method serializes the object's data, which is converted into a dictionary by `get_as_dict()`,
        and saves it to a pickle file. The file is saved within a designated subdirectory in `_save_path`.
        The filename is constructed from the object's set name (`_set_name`), converted to uppercase, and
        appended with `_data_dict.pkl`.

        The method first checks if the target directory exists within `_save_path`; if not, it creates it.
        The data is then saved using Python's `pickle` module, which serializes the dictionary into a binary
        format.

        Attributes
        ----------
        _save_path : str
            The base directory path where the data directory will be created and the pickle file saved.
        _set_name : str
            The base name used to construct the filename of the pickle file.

        Returns
        -------
        None
        """
        data_path = os.path.join(self._save_path, "data")

        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        full_file_path = os.path.join(data_path, f"{self._set_name.upper()}_data_dict.pkl")
        data_dict = self.get_as_dict()

        with open(full_file_path, 'wb') as file:
            pickle.dump(data_dict, file)

    def get_predictions(self):
        """
        Calculate and return the predictions and their corresponding true labels
        from accumulated data over an epoch.

        This method processes tensor data stored in `self.epoch_y_pred` and `self.epoch_y_true`,
        converting them from PyTorch tensors to NumPy arrays. If the model is binary classification
        (i.e., `_num_classes` is 1), it handles tensors individually, moving them to CPU memory before conversion.
        For multi-class models, it converts the entire list directly.

        Returns
        -------
        y_pred_numpy : ndarray
            A NumPy array containing the predicted labels for the dataset.
        y_true_numpy : ndarray
            A NumPy array containing the true labels for the dataset.

        Notes
        -----
        This method assumes that `self.epoch_y_pred` and `self.epoch_y_true` are lists of PyTorch tensors.
        These tensors should be on the same device and in the appropriate format before calling this method.
        """
        if self._num_classes == 1:
            y_pred_numpy = np.array([tensor.cpu().numpy() for tensor in self.epoch_y_pred])
            y_true_numpy = np.array([tensor.cpu().numpy() for tensor in self.epoch_y_true])
        else:
            y_pred_numpy = np.array(self.epoch_y_pred)
            y_true_numpy = np.array(self.epoch_y_true)

        return y_pred_numpy, y_true_numpy

    def save_to_mlflow(self):
        """
        Logs various performance metrics of a machine learning model to MLflow.

        This method iterates over a predefined list of metric names, retrieves the metric values
        stored as attributes of the instance, and logs them to MLflow. Each metric is prefixed
        with the set name (e.g., training or validation) to categorize the metrics in MLflow.

        It assumes that each metric is stored as a list of values, with each value corresponding
        to a specific epoch, and logs the metric value for each epoch.

        Notes
        -----
        - This method should only be called if MLflow is set up properly in the environment.
        - Metrics are expected to be stored in attributes named with a leading underscore followed
          by the metric name (e.g., `_accuracy` for accuracy values).
        - This method does not handle cases where metric lists are of different lengths or do not
          exist, beyond checking for emptiness.

        Raises
        ------
        AttributeError
            If any expected metric attribute does not exist or is not accessible.
        """
        epochs = len(self._accuracy)

        metric_names = ["loss", "accuracy", "precision", "recall", "f1", "auc", "kappa", "mcc"]
        for epoch in range(epochs):
            for metric_name in metric_names:
                metric_value_list = getattr(self, f"_{metric_name}")
                if metric_value_list:  # Check if the metric list is not empty
                    mlflow.log_metric(f"{self._set_name}_{metric_name}", metric_value_list[epoch], step=epoch)


class MCHistory:
    def __init__(self, save_path, num_classes, name="MCHistory"):
        self._save_path: str = save_path
        self._fig_path: str = check_folder(save_path, path_ext="figures")
        self._set_name: str = name
        self._num_classes: int = num_classes

        self._labels = []
        self._all_forward_predictions = []

        self._mean_predictions: Optional[np.ndarray] = None
        self._predictive_entropy: Optional[np.ndarray] = None
        self._variance: Optional[np.ndarray] = None
        self._true_class_variance: Optional[np.ndarray] = None
        self._mutual_information: Optional[np.ndarray] = None

        self._accuracy: Optional[np.ndarray] = None
        self._ensemble_accuracy: Optional[float] = None

        # Figure relevant
        self.fig_size = (20, 12)
        self.dpi = 300
        self.size_of_font = 20

    @property
    def ensemble_accuracy(self):
        return self._ensemble_accuracy

    @property
    def mean_predictions(self):
        return self._mean_predictions

    @property
    def get_prediction_set(self):
        return np.array(self.mean_predictions), np.array(self._labels)

    def on_pass_end(self, predictions, labels):
        """
        Handle the actions to be performed at the end of each prediction pass.

        This method updates the internal storage for labels and predictions after each prediction pass.
        It initializes the `_labels` attribute with the first set of labels if they haven't been stored yet,
        and it appends each set of predictions to `_all_forward_predictions`.

        Parameters
        ----------
        predictions : array_like
            The predictions made during the current pass. This can be a list or a numpy array of predictions,
            typically shaped as (n_samples, n_classes) representing the probability or output scores
            for each class.
        labels : array_like
            The true labels corresponding to the predictions. This should match the format and length
            of `predictions`. It is typically a one-hot encoded numpy array or a list of label indices.

        Attributes
        ----------
        _labels : ndarray
            A numpy array storing the true labels. If `_labels` is initially empty, it will be set during
            the first pass and not modified thereafter.
        _all_forward_predictions : list
            A list that accumulates the predictions from each pass. Each element in the list is the `predictions`
            array from a respective pass.

        Notes
        -----
        This method is designed to be called at the end of each forward pass to accumulate predictions and
        initialize label data if not already done.

        Returns
        -------
        None
        """
        if len(self._labels) == 0:
            self._labels = np.array(labels)
        self._all_forward_predictions.append(predictions)

    def calculate_metrics(self):
        """
        Calculate various statistical metrics for model evaluation and visualization.

        This method orchestrates the computation of mean predictions, variance, predictive entropy,
        accuracy, and mutual information from model predictions. It also generates histograms and plots
        correlating these metrics with accuracy. This requires that both labels and predictions are
        already loaded into the object.

        Raises
        ------
        ValueError
            If either `_labels` or `_all_forward_predictions` is None, indicating that necessary
            prediction data has not been loaded prior to metric calculation.

        Notes
        -----
        - This method assumes that `get_mc_prediction` function (or equivalent) has been called to populate
          `_labels` and `_all_forward_predictions` with the necessary data.
        - The method internally converts `_all_forward_predictions` from a list (if necessary) to a numpy array
          for consistent processing across various metric calculations.
        - Various plotting methods are called to visualize the metrics, which should be defined elsewhere in
          the class:
            - `plot_histogram` for distribution visualization of predictive entropy and mutual information.
            - `plot_metric_vs_accuracy` to show how predictive entropy, mutual information, and variance correlate
              with model accuracy.

        Steps performed:
        - Conversion of `_all_forward_predictions` to a numpy array if not already.
        - Calculation of mean predictions, variance, predictive entropy, accuracy, and mutual information.
        - Visualization of metrics through histograms and scatter plots.

        Returns
        -------
        None
        """
        if self._labels is None or self._all_forward_predictions is None:
            raise ValueError("Labels or predictions are None, make sure that the get_mc_prediction function is called")
        self._all_forward_predictions = np.array(self._all_forward_predictions)

        # Compute the necessary statistical metrics
        self._calculate_mean_predictions()
        self._calculate_variance()
        self._calculate_predictive_entropy()
        self._calculate_accuracy()
        self._calculate_mutual_information()

        # Generate visualizations for the computed metrics
        self.plot_histogram(self._predictive_entropy, 'Predictive Entropy', color="darkblue")
        self.plot_histogram(self._mutual_information, 'Mutual Information', color="darkorange")
        self.plot_metric_vs_accuracy(metric_data=self._predictive_entropy, metric_name='Predictive Entropy')
        self.plot_metric_vs_accuracy(metric_data=self._mutual_information, metric_name='Mutual Information')
        self.plot_metric_vs_accuracy(metric_data=self._true_class_variance, metric_name='Variance')

        # todo Check the performance of the model vs prediction and labels

    def _calculate_mean_predictions(self):
        """
        Calculate the mean of all predictions across multiple prediction passes.

        This method computes the average predictions from the stored prediction data in
        `_all_forward_predictions`. The mean is calculated over the axis representing
        different prediction passes, resulting in an averaged prediction for each sample
        across all classes. The result is stored in `_mean_predictions`.

        Notes
        -----
        - `_all_forward_predictions` should be a list of numpy arrays or a single numpy array
          where each element (or row if a single array) corresponds to predictions from one
          prediction pass. The dimensions of `_all_forward_predictions` should be structured as
          (n_passes, n_samples, n_classes).
        - This method updates the `_mean_predictions` attribute of the class instance with the
          computed mean across the specified axis.

        Returns
        -------
        None
        """
        self._mean_predictions = np.mean(self._all_forward_predictions, axis=0)

    def _calculate_variance(self):
        """
        Calculate the variance across prediction passes and the variance for the true class predictions.

        This method computes the variance of predictions across different forward passes stored
        in `_all_forward_predictions`. It then determines the variance of the predicted values
        specifically for the true class labels indicated by `_labels`.

        Attributes
        ----------
        _variance : ndarray
            Variance of predictions across different forward passes for each class.
        _true_class_variance : ndarray
            Variance of predictions for the actual classes of the samples.

        Notes
        -----
        - `_all_forward_predictions` should be a numpy array where each row corresponds to a different
          prediction pass and each column to a different class. The variance is calculated across the rows.
        - `_labels` should be a one-hot encoded numpy array of the true labels.
        - This method updates the `_variance` and `_true_class_variance` attributes of the class instance.

        Example
        -------
        Assuming `_all_forward_predictions` is an array of shape (n_passes, n_samples, n_classes)
        and `_labels` is a one-hot encoded array of shape (n_samples, n_classes), this method
        calculates the variance across the `n_passes` for each class and sample, and extracts the variance
        for the true class of each sample.

        Returns
        -------
        None
        """
        self._variance = np.var(self._all_forward_predictions, axis=0)
        indices = np.argmax(self._labels, axis=1)
        self._true_class_variance = np.array([self._variance[i, label] for i, label in enumerate(indices)])

    @staticmethod
    def _entropy(probs):
        """
        Calculate the Shannon entropy of a probability distribution for each set of probabilities in an array.

        Shannon entropy is a measure of the unpredictability or the randomness of a probability distribution.
        This method applies a small constant to the probabilities to avoid mathematical issues with logarithms
        of zero.

        Parameters
        ----------
        probs : ndarray
            An array of probability distributions where each row corresponds to a distinct distribution.
            Each distribution (row) should sum to approximately one.

        Returns
        -------
        ndarray
            An array containing the Shannon entropy values for each probability distribution provided in `probs`.

        Notes
        -----
        The entropy is calculated using the formula:
        - ∑(p * log(p + small_constant)), where `p` are the probabilities in `probs` and `small_constant`
        is a small value added to prevent undefined logarithmic calculations when `p` includes zero.
        """
        small_constant = 1e-10  # to prevent log(0)
        return -np.sum(probs * np.log(probs + small_constant), axis=1)

    def _calculate_mutual_information(self):
        """
        Calculate the mutual information between the ensemble predictions and the true labels.

        This method computes mutual information as the difference between the predictive entropy
        and the expected entropy of the model's predictions. The predictive entropy should be
        pre-computed and stored in `_predictive_entropy`. The expected entropy is calculated as
        the mean entropy across all prediction passes stored in `_all_forward_predictions`.

        Raises
        ------
        ValueError
            If `_predictive_entropy` is not already calculated and set.

        Notes
        -----
        This method modifies the internal state by updating the `_mutual_information` attribute
        of the class instance.

        Attributes such as `_predictive_entropy` and `_all_forward_predictions` are expected to be
        initialized and populated with relevant data before this method is called. `_all_forward_predictions`
        should contain individual prediction arrays for each pass, and `_entropy` method is used to
        calculate entropy for each of these prediction arrays.

        Example usage of this method assumes that you have a method `_entropy` defined in your class
        that calculates the entropy of a given set of predictions.

        Returns
        -------
        None
        """
        if self._predictive_entropy is None:
            raise ValueError("Predictive entropy has to be calculated first!")
        # Calculate the expected entropy of the predictions: Mean of the entropies of each pass's predictions
        individual_entropies = np.array([self._entropy(pred) for pred in self._all_forward_predictions])
        expected_entropy = np.mean(individual_entropies, axis=0)

        # Calculate mutual information: Difference between predictive entropy and expected entropy
        mutual_information = self._predictive_entropy - expected_entropy
        self._mutual_information = mutual_information

    def _calculate_accuracy(self):
        """
        Calculate the accuracy of ensemble predictions by comparing them with the true labels.

        This method computes the accuracy by determining the percentage of predictions
        that match the true labels. It relies on the following attributes:
        - `_mean_predictions`: numpy array of predicted probabilities for each class per sample.
        - `_labels`: numpy array of actual labels in a one-hot encoded format.

        The accuracy is stored in the `_accuracy` attribute as a boolean array where each element
        corresponds to whether a prediction was correct. The overall ensemble accuracy,
        calculated as the percentage of correct predictions, is stored in the `_ensemble_accuracy`
        attribute.

        Notes
        -----
        This method modifies the internal state by updating the `_accuracy` and `_ensemble_accuracy`
        attributes of the class instance.

        It is expected that `_mean_predictions` and `_labels` are already initialized and
        formatted correctly (i.e., both are numpy arrays, `_mean_predictions` contains probabilities
        across columns representing classes, and `_labels` is one-hot encoded).
        """
        predictions = np.argmax(self._mean_predictions, axis=1)
        labels = np.argmax(self._labels, axis=1)
        self._accuracy = np.array(predictions == labels)
        self._ensemble_accuracy = (np.sum(self._accuracy) / len(labels)) * 100

    def _calculate_predictive_entropy(self):
        """
        Calculate the predictive entropy of predictions from multiple stochastic forward passes.

        Returns
        -------
        numpy.ndarray
            An array containing the predictive entropy for each sample.
        """
        if self._mean_predictions is None:
            raise ValueError("Mean predictions is None...")
        # Compute the entropy for each sample
        self._predictive_entropy = self._entropy(probs=self._mean_predictions)

    def get_as_dict(self):
        """
        Return the attributes of the model as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the model's data and metrics, including labels, predictions,
            uncertainty measures, and accuracy metrics.

        Notes
        -----
        This method provides a structured way to access all significant attributes stored within the
        model instance, which can be useful for serialization, debugging, or data analysis.
        """
        return {
            'labels': self._labels,
            'all_forward_predictions': self._all_forward_predictions,
            'mean_predictions': self._mean_predictions,
            'predictive_entropy': self._predictive_entropy,
            'variance': self._variance,
            'true_class_variance': self._true_class_variance,
            'mutual_information': self._mutual_information,
            'accuracy': self._accuracy,
            'ensemble_accuracy': self._ensemble_accuracy
        }

    def save_to_pickle(self):
        """
        Save the object's data in a dictionary format to a pickle file.

        This method serializes the object's data, which is converted into a dictionary by `get_as_dict()`,
        and saves it to a pickle file. The file is saved within a designated subdirectory in `_save_path`.
        The filename is constructed from the object's set name (`_set_name`), converted to uppercase, and
        appended with `_data_dict.pkl`.

        The method first checks if the target directory exists within `_save_path`; if not, it creates it.
        The data is then saved using Python's `pickle` module, which serializes the dictionary into a binary
        format.

        Attributes
        ----------
        _save_path : str
            The base directory path where the data directory will be created and the pickle file saved.
        _set_name : str
            The base name used to construct the filename of the pickle file.

        Returns
        -------
        None
        """
        data_path = os.path.join(self._save_path, "data")

        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        full_file_path = os.path.join(data_path, f"{self._set_name.upper()}_data_dict.pkl")
        data_dict = self.get_as_dict()

        with open(full_file_path, 'wb') as file:
            pickle.dump(data_dict, file)

    def plot_histogram(self, metric_data, metric_name, color='darkblue'):
        """
        Plot a histogram of either predictive entropy or mutual information.

        Parameters:
        - data: numpy.ndarray, the array of predictive entropy or mutual information values.
        - name: str, the name of the measurement ('Predictive Entropy' or 'Mutual Information').
        """
        plt.figure(figsize=self.fig_size, dpi=self.dpi)
        sns.histplot(metric_data, bins=30, alpha=0.75, color=color)
        plt.title(f'Histogram of {metric_name}', fontsize=self.size_of_font)
        plt.xlabel(metric_name, fontsize=self.size_of_font)
        plt.ylabel('Frequency', fontsize=self.size_of_font)
        plt.tick_params(axis='both', which='major', labelsize=20)

        if metric_name == 'Predictive Entropy':
            # Maximum entropy line, only relevant for predictive entropy
            plt.axvline(x=np.log(self._num_classes), color='red', linestyle='--',
                        label=f'Max Entropy: {np.log(self._num_classes):.2f}')

            plt.legend(fontsize=self.size_of_font)
        self._save_or_show(filename=f"histogram_{metric_name.strip()}")
    
    def plot_metric_vs_accuracy(self, metric_data, metric_name):
        """
        Create a scatter plot of a given metric against accuracy.

        Parameters:
        - metric_data : numpy.ndarray, the array containing the metric values to plot.
        - metric_name : str, the name of the metric (e.g., 'Predictive Entropy').
        - accuracy : numpy.ndarray, a boolean array indicating whether each prediction was correct.
        """
        plt.figure(figsize=self.fig_size, dpi=self.dpi)

        # Boolean indexing to separate correct and incorrect predictions
        correct_indices = self._accuracy
        incorrect_indices = ~self._accuracy

        # Plotting correct predictions
        plt.scatter(metric_data[correct_indices],
                    np.random.rand(np.sum(correct_indices)),  # Random y-values for visibility
                    color='green',
                    label='Correct',
                    alpha=0.6,
                    edgecolors='w',
                    s=100)  # Larger scatter points

        # Plotting incorrect predictions
        plt.scatter(metric_data[incorrect_indices],
                    np.random.rand(np.sum(incorrect_indices)),  # Random y-values for visibility
                    color='red',
                    label='Incorrect',
                    alpha=0.6,
                    edgecolors='w',
                    s=100)  # Larger scatter points

        plt.xlabel(metric_name, fontsize=self.size_of_font)
        plt.yticks([])  # Hide y-axis ticks since they're not informative
        plt.title(f'{metric_name} vs. Prediction Accuracy', fontsize=self.size_of_font)
        plt.tick_params(axis='x', which='major', labelsize=self.size_of_font)

        if metric_name == 'Predictive Entropy':
            # Maximum entropy line, only relevant for predictive entropy
            plt.axvline(x=np.log(self._num_classes), color='red', linestyle='--',
                        label=f'Max Entropy: {np.log(self._num_classes):.2f}')
        plt.legend(fontsize=self.size_of_font)
        self._save_or_show(filename=f"accuracy_vs_{metric_name.strip()}")

    def _save_or_show(self, filename):
        """
            Save the currently active matplotlib plot to a PDF file or display it on screen.

            This method decides whether to save the current matplotlib plot based on the presence
            of a directory path in `_fig_path`. If `_fig_path` is set, the plot is saved as a PDF
            file in the specified directory with the given filename. If `_fig_path` is None,
            the plot is displayed using `plt.show()`. After saving or showing the plot, it is closed
            using `plt.close()` to free up memory.

            Parameters
            ----------
            filename : str
                The name of the file to save the plot as, without the extension. Used only if `_fig_path`
                is not None.

            Attributes
            ----------
            _fig_path : str or None
                The directory path where the plot should be saved. If None, the plot is displayed on screen.

            Returns
            -------
            None
            """
        if self._fig_path:
            full_path = f"{self._fig_path}/{filename}.pdf"
            plt.savefig(full_path, format="pdf")
            mlflow.log_artifact(full_path)
        else:
            plt.show()
        plt.close()

    def save_to_mlflow(self):
        mlflow.log_metric()
