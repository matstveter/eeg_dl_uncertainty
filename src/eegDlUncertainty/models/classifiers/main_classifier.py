import abc
import json
import os.path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import CosineAnnealingLR

from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.models.classifiers.inceptionTime import InceptionNetwork
from eegDlUncertainty.models.classifiers.swag import SWAG, bn_update
from eegDlUncertainty.models.get_models import get_models


class MainClassifier(abc.ABC, nn.Module):
    def __init__(self, model_name: str, pretrained: Optional[str] = None, **kwargs):
        super().__init__()

        # Value used for temperature scaling...
        self.temperature = torch.nn.Parameter(torch.ones(1))

        if pretrained is not None:
            self.classifier = self.from_disk(path=pretrained)
        else:
            kwargs['classifier_name'] = model_name
            self.classifier = get_models(model_name=model_name, **kwargs)
            hyperparameters = kwargs.copy()
            self._hyperparameters = hyperparameters
            self._name = model_name

            self.save_path: str = kwargs.get("save_path")

            if self.save_path is not None:
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
                  earlystopping_patience: int, **kwargs):

        best_loss = 1_000_000
        optimizer_name = kwargs.pop("optimizer_name", "adam")

        # Set default parameters and update from kwargs if available
        if optimizer_name == "sgd":
            lr = kwargs.pop("lr", self._learning_rate)
            momentum = kwargs.pop("momentum", 0.9)
            nesterov = kwargs.pop("nesterov", True)
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

        elif optimizer_name == "adam":
            lr = kwargs.pop("lr", self._learning_rate)
            betas = kwargs.pop("betas", (0.9, 0.999))
            eps = kwargs.pop("eps", 1e-08)
            weight_decay = kwargs.pop("weight_decay", 0)
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        elif optimizer_name == "nadam":
            lr = kwargs.pop("lr", self._learning_rate)
            betas = kwargs.pop("betas", (0.9, 0.999))
            eps = kwargs.pop("eps", 1e-08)
            weight_decay = kwargs.pop("weight_decay", 0)
            optimizer = torch.optim.NAdam(self.classifier.parameters(), lr=lr, betas=betas, eps=eps,
                                          weight_decay=weight_decay)

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)

        best_path = None

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
                best_path = os.path.join(self._model_path, f"{self._name}_model")
                self.classifier.save(path=best_path)
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

        if best_path is not None:
            # Set the current model to the best model during training
            self.classifier = self.classifier.load(path=best_path)

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

    def set_temperature(self, val_loader, criterion, device, model_name=None):
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
        self.temperature.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.001, max_iter=1000)

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
        print(f"Optimal temperature {model_name}", self.temperature.item())

    def get_predictions(self, loader, device, get_prob=False):
        """
        Get predictions from the model for the given data loader.

        This method applies the model to the data from the loader and collects the predictions
        and ground truth labels. The model is set to evaluation mode and computations are performed
        without gradient tracking.

        Parameters
        ----------
        loader : DataLoader
            The DataLoader providing the data for prediction. It should yield batches of
            data and corresponding labels.
        device : torch.device
            The device to which the data should be transferred. This is typically
            a CUDA device or CPU.

        Returns
        -------
        tuple of np.array
            A tuple containing two numpy arrays. The first array contains the model's
            predictions for each input sample. The second array contains the ground truth
            labels for each input sample.

        Notes
        -----
        The method moves the data to the specified device before applying the model.
        The model's predictions are obtained by calling the `predict_prob` method.
        The predictions and labels are collected in lists, which are then converted to numpy arrays.
        """
        self.eval()
        preds, ground_truth = [], []
        with torch.no_grad():
            for inp, lab in loader:
                inputs, label = inp.to(device), lab.to(device)
                if get_prob:
                    outputs = self.predict_prob(inputs)
                else:
                    outputs = self(inputs)
                preds.extend(outputs.cpu().numpy())
                ground_truth.extend(label.cpu().numpy())
        return np.array(preds), np.array(ground_truth)

    def save_model(self, path):
        self.classifier.save(path=path)


class MCClassifier(MainClassifier):

    def forward_ensemble(self, x: torch.Tensor, num_sampling=50):
        with torch.no_grad():
            self.eval()
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

            # Sample the model num_sampling times
            logits = []
            for _ in range(num_sampling):
                # Append the logits to a list
                logits.append(self(x))

        return torch.stack(logits)


class SnapshotClassifier(MainClassifier):
    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History,
                  earlystopping_patience: int, **kwargs):
        print(kwargs)
        start_lr = kwargs.pop("start_lr")
        num_cycles = kwargs.pop("num_cycles")
        epochs_per_cycle = kwargs.pop("epochs_per_cycle")
        use_best = kwargs.pop("use_best")

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=start_lr)
        self.to(device)
        model_weight_paths = []
        for cycle in range(num_cycles):
            # Manually reset the learning rate for the optimizer without resetting its state
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr

            scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_cycle, eta_min=0.000001)
            best_loss = float('inf')
            best_model_path = None

            print(f"Cycle {cycle + 1}/{num_cycles}")
            self.train()
            for epoch in range(epochs_per_cycle):
                for data, targets in train_loader:
                    inputs, targets = data.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = loss_fn(outputs, targets)

                    y_pred = self.activation_function(logits=outputs)
                    train_hist.batch_stats(y_pred=y_pred, y_true=targets, loss=loss)

                    loss.backward()
                    optimizer.step()

                train_hist.on_epoch_end()
                scheduler.step()
                print(f"\n--- CYCLE: {cycle + 1} / {num_cycles} EPOCH {epoch + 1} / {epochs_per_cycle} --- LR: "
                      f"{scheduler.get_last_lr()[0]}")

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

                if val_history.get_last_loss() < best_loss and use_best:
                    best_loss = val_history.get_last_loss()
                    best_model_path = os.path.join(self._model_path, f"snapshot_{cycle}_model")
                    self.classifier.save(path=best_model_path)

            if best_model_path and use_best:
                # save the abs path to the models making it easier to load it later...
                model_weight_paths.append(best_model_path)
            else:
                path = os.path.join(self._model_path, f"snapshot_{cycle}_model")
                self.classifier.save(path=path)
                model_weight_paths.append(path)

        return model_weight_paths


class FGEClassifier(MainClassifier):

    @staticmethod
    def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
        t = (epoch % cycle) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: object, val_history: object,
                  earlystopping_patience: int, **kwargs):

        pretrain_epochs = kwargs.pop("fge_start_epoch")
        epochs_per_cycle = kwargs.pop("fge_epochs_per_cycle")
        cycle_start_lr = kwargs.pop("fge_cycle_start_lr")
        cycle_end_lr = kwargs.pop("fge_cycle_end_lr")
        num_models = kwargs.pop("fge_num_models")

        num_epochs = pretrain_epochs + (epochs_per_cycle * num_models)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)
        model_weight_paths = []

        # Pretraining Phase
        for epoch in range(pretrain_epochs):
            print(f"\n--- PRETRAIN EPOCH {epoch + 1} / {pretrain_epochs} ---")
            self.train()
            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)

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

                    y_pred = self.activation_function(logits=outputs)
                    val_history.batch_stats(y_pred=y_pred, y_true=targets, loss=val_loss)
                val_history.on_epoch_end()

        # Fast-Geometric Ensemble Phase
        for epoch in range(pretrain_epochs, num_epochs):
            lr = self.cyclic_learning_rate(epoch=epoch - pretrain_epochs, cycle=epochs_per_cycle,
                                           alpha_1=cycle_start_lr, alpha_2=cycle_end_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"\n--- EPOCH {epoch + 1} / {num_epochs} --- LR: {lr}")
            self.train()
            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)

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

                    y_pred = self.activation_function(logits=outputs)
                    val_history.batch_stats(y_pred=y_pred, y_true=targets, loss=val_loss)
                val_history.on_epoch_end()

            # At halfway through the cycle, save the model weights
            if (epoch - pretrain_epochs) % epochs_per_cycle == epochs_per_cycle // 2:
                path = os.path.join(self._model_path, f"FGE_cycle_{epoch}_model")
                self.classifier.save(path=path)
                model_weight_paths.append(path)
                print("\nModel saved at epoch", epoch + 1, " lr_schedule: ", lr)

        return model_weight_paths


class SWAGClassifier(MainClassifier):

    def __init__(self, model_name, pretrained: Optional[str] = None, **kwargs):
        model_kwargs = kwargs.copy()
        model_kwargs['classifier_name'] = model_name
        super().__init__(model_name=model_name, pretrained=pretrained, **kwargs)
        self.swag_model = SWAG(base=InceptionNetwork, max_num_models=kwargs.pop("swag_num_models"),
                               no_cov_mat=False, **model_kwargs)

    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History,
                  earlystopping_patience: int, **kwargs):

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)

        swag_start = kwargs.get("swag_start")
        swag_freq = kwargs.get("swag_freq")
        swag_lr = kwargs.get("swag_lr")
        swag_num_models = kwargs.get("swag_num_models")

        num_epochs = swag_start + (swag_freq * swag_num_models)

        for epoch in range(num_epochs):

            print(f"\n-------------------------  EPOCH {epoch + 1} / {num_epochs}  -------------------------")
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

            # Change to a SGD optimizer with a different learning rate for SWAG training
            if epoch == swag_start:
                print("Changing optimizer to SGD for SWAG training")
                optimizer = torch.optim.SGD(self.classifier.parameters(), lr=swag_lr,
                                            weight_decay=1e-4)

            if epoch >= swag_start and (epoch - swag_start) % swag_freq == 0:
                print("Collecting model for SWAG")
                self.swag_model.collect_model(self.classifier)

        print("Testing the SWAG model")
        # path = os.path.join(self._model_path, f"{self._name}_last_model")
        self.swag_model.sample(0.0)
        bn_update(train_loader, self.swag_model, device=device)
        # self.swag_model.save(path=path)
        self.save_hyperparameters()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.swag_model(inputs)

                val_loss = loss_fn(outputs, targets)

                # Activation function and store values
                y_pred = self.activation_function(logits=outputs)
                val_history.batch_stats(y_pred=y_pred, y_true=targets, loss=val_loss)
            val_history.on_epoch_end()

    def forward_ensemble(self, x: torch.Tensor, num_sampling=50):
        self.eval()  # Ensure the model is in evaluation mode
        logits = []

        with torch.no_grad():  # Disable gradient calculation
            for _ in range(num_sampling):
                self.swag_model.sample()  # Sample from the SWAG model
                logits.append(self.swag_model(x))  # Append the logits to the list
        return torch.stack(logits)  # Stack the logits into a single tensor

    def save_swag_model(self):
        # todo This function need to save swag with all parameters
        pass
