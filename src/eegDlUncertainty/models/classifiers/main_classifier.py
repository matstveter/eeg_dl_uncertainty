import abc
import json
import os.path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.models.get_models import get_models


class MainClassifier(abc.ABC, nn.Module):
    def __init__(self, model_name: str, pretrained: Optional[str] = None, **kwargs):
        super().__init__()

        # Value used for temperature scaling...
        self.temperature = torch.nn.Parameter(torch.ones(1))

        if pretrained is not None:
            self.classifier = self.from_disk(path=pretrained)
        else:
            self.classifier = get_models(model_name=model_name, **kwargs)
            hyperparameters = kwargs.copy()
            # hyperparameters['classifier_name'] = model_name
            self._hyperparameters = hyperparameters
            self._name = model_name

            save_path: str = kwargs.get("save_path")

            if save_path is not None:
                self._model_path = os.path.join(kwargs.get("save_path"), "model")
            else:
                raise ValueError("Save path not specified!")
            if not os.path.exists(self._model_path):
                os.makedirs(self._model_path, exist_ok=True)

            self._learning_rate = kwargs.get("learning_rate")

    @property
    def hyperparameters(self):
        return self._hyperparameters

    def model_path(self, with_ext=True):
        if with_ext:
            return os.path.join(self._model_path, f"{self._name}_model")
        else:
            return self._model_path

    def save_hyperparameters(self):
        hyper_param = self.classifier.hyperparameters
        with open(os.path.join(f"{self._model_path}/hyper_param_dict.json"), "w") as file:
            json.dump(hyper_param, file)

    def forward(self, x: torch.Tensor, **kwargs):
        """ Forward function calling the self.classifiers forward function, in addition in divides the produced logits
        by the self.temperature value. If this value is not optimized it will only be 1.

        Returns
        -------
        temperature scaled predictions if self.temperature is trained

        """
        logits = self.classifier(x, **kwargs)
        return logits / self.temperature

    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability predictions for a given input tensor.

        This method applies the model to the input tensor `x` to compute logits,
        then applies an activation function to obtain the probability predictions.
        The calculation is done without gradient tracking.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing data for prediction. The shape and data type
            of `x` should be compatible with the model's expected input.

        Returns
        -------
        torch.Tensor
            A tensor containing the probability predictions for each input sample.
            The output tensor shape depends on the model configuration and the
            activation function used.

        """
        with torch.no_grad():
            prediction = self.activation_function(logits=self(x))
        return prediction

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the class predictions for a given input tensor.

        This method applies the model to the input tensor `x` to compute logits,
        then applies an activation function with `ret_prob=False` to obtain class
        predictions. The calculation is done without gradient tracking. This
        is suitable for scenarios where class labels are needed instead of
        probabilities.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing data for prediction. The shape and data type
            of `x` should be compatible with the model's expected input.

        Returns
        -------
        torch.Tensor
            A tensor containing the class predictions for each input sample. The
            output tensor's shape and data type depend on the model configuration
            and the activation function used. Typically, it contains integer indices
            representing the predicted classes.

        """
        with torch.no_grad():
            prediction: torch.Tensor = self.activation_function(logits=self(x), ret_prob=False)
        return prediction

    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History,
                  earlystopping_patience: int):

        best_loss = 1_000_000
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)

        # For earlystopping
        epochs_with_no_improvement = 0
        early_stop = False
        for epoch in range(training_epochs):
            if early_stop:
                break

            print(f"\n-------------------------  EPOCH {epoch + 1} / {training_epochs}  -------------------------")
            self.train()

            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                # Store values in the history object
                y_pred = self.activation_function(logits=outputs)
                train_hist.batch_stats(y_pred=y_pred, y_true=targets, loss=loss)

                loss.backward()
                optimizer.step()

            train_hist.on_epoch_end()

            self.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self(inputs)

                    val_loss = loss_fn(outputs, targets)

                    # Activation function and store values
                    y_pred = self.activation_function(logits=outputs)
                    val_history.batch_stats(y_pred=y_pred, y_true=targets, loss=val_loss)
                val_history.on_epoch_end()

            if val_history.get_last_loss() < best_loss:
                best_loss = val_history.get_last_loss()
                path = os.path.join(self._model_path, f"{self._name}_model")
                self.classifier.save(path=path)
                epochs_with_no_improvement = 0
            else:
                epochs_with_no_improvement += 1

            if 0 < earlystopping_patience <= epochs_with_no_improvement:
                early_stop = True
                mlflow.log_metric('earlystopping_inferred', (epoch + 1))
                print(f"\nStopping early at epoch {epoch + 1}!")

        path = os.path.join(self._model_path, f"{self._name}_last_model")
        self.classifier.save(path=path)
        self.save_hyperparameters()

    def test_model(self, *, test_loader: DataLoader, device: torch.device, test_hist: History, loss_fn: _Loss):
        self.to(device)
        with torch.no_grad():
            self.eval()
            for inputs, targets in test_loader:
                inputs, target = inputs.to(device), targets.to(device)

                outputs = self(inputs)
                loss = loss_fn(outputs, target)

                y_pred = self.activation_function(outputs)
                test_hist.batch_stats(y_pred=y_pred, y_true=target, loss=loss)
            test_hist.on_epoch_end()

    @staticmethod
    def activation_function(logits, ret_prob=True):
        """
        Applies an activation function to the logits based on their shape.

        This method applies a softmax activation function if the last dimension
        of `logits` is 3, indicating a multi-class classification problem. Otherwise,
        it applies a sigmoid activation function, assuming a binary classification
        problem. It can return either probabilities or class labels based on the
        `ret_prob` flag.

        Parameters
        ----------
        logits : torch.Tensor
            The input tensor containing logits from a model's output. The shape of
            `logits` determines which activation function is applied.
        ret_prob : bool, optional
            A flag determining the type of output. If `True` (default), the method
            returns probabilities. If `False`, it returns class labels. For softmax,
            class labels are the indices of the max probability. For sigmoid, labels
            are obtained by rounding the probabilities.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the activation function. If `ret_prob`
            is `True`, it contains probabilities. If `False`, it contains class labels
            as integers for multi-class or binaries for binary classification problems.

        Notes
        -----
        - For multi-class classification (softmax), the output tensor has the same shape
          as the input if `ret_prob` is `True`. If `ret_prob` is `False`, the output tensor
          shape will have one less dimension, representing the class label with the highest
          probability for each input.
        - For binary classification (sigmoid), the output tensor always has the same shape
          as the input, with each element representing the probability or binary class label.

        """
        if logits.shape[1] == 3:
            outp = torch.softmax(logits, dim=1)
            if not ret_prob:
                _, outp = torch.max(outp, dim=1)
        else:
            outp = torch.sigmoid(logits)
            if not ret_prob:
                outp = torch.round(outp)
        return outp

    def from_disk(self, path: str):
        # Get state
        state = torch.load(path)

        # Initialise model
        model = get_models(model_name=state["classifier_name"], **state["hyperparameters"])
        self._hyperparameters = state['hyperparameters']
        self._name = state['classifier_name']

        # Load parameters
        model.load_state_dict(state_dict=state["state_dict"], strict=True)

        return model

    def set_temperature(self, val_loader, criterion, device, model_name):
        """
        Calibrates the model's temperature parameter using the provided validation loader.

        This method sets the model to evaluation mode and moves it to the specified device.
        It then optimizes the temperature parameter to minimize the loss computed by the
        given criterion on the validation dataset. The temperature is optimized using the
        LBFGS optimizer. The optimal temperature is logged using mlflow.

        Parameters
        ----------
        model_name
        val_loader : DataLoader
            The DataLoader providing the validation dataset. It should yield batches of
            data and corresponding labels.
        criterion : _Loss
            The loss function used to evaluate the model's predictions. It should be
            compatible with the model's output and the labels provided by `val_loader`.
        device : torch.device
            The device to which the model and data should be transferred. This is typically
            a CUDA device or CPU.

        Notes
        -----
        - The method logs the optimal temperature found during optimization using mlflow,
          under the metric name "Optimal temperature".
        - The optimization is performed without gradient tracking, and the method prints
          the loss after each batch is processed.

        """
        self.eval()
        self.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=5000)

        def evaluation():
            loss = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)

                    logits = self.forward(data)

                    loss += criterion(logits, labels).item()
            return loss / len(val_loader)

        optimizer.step(lambda: -evaluation())
        mlflow.log_metric(f"Optimal temperature {model_name}", self.temperature.item())
