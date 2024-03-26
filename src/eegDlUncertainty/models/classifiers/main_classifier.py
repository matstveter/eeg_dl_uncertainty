import abc
import json
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from eegDlUncertainty.data.results.history import History
from eegDlUncertainty.models.get_models import get_models


class MainClassifier(abc.ABC, nn.Module):
    def __init__(self, model_name: str, pretrained=None, **kwargs):
        super().__init__()
        if pretrained is not None:
            self.classifier = self.from_disk(path=pretrained)
        else:
            self.classifier = get_models(model_name=model_name, **kwargs)
            hyperparameters = kwargs.copy()
            # hyperparameters['classifier_name'] = model_name
            self._hyperparameters = hyperparameters
            self._name = model_name

            self._model_path = os.path.join(kwargs.get("save_path"), "model")
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
        return self.classifier(x, **kwargs)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prediction = self.activation_function(logits=self(x))
        return prediction

    def fit_model(self, *, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int,
                  device: torch.device, loss_fn: _Loss, train_hist: History, val_history: History):

        best_loss = 1_000_000
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self._learning_rate)
        self.to(device)
        for epoch in range(training_epochs):
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

        self.save_hyperparameters()

    def test_model(self, *, test_loader, device, test_hist, loss_fn: _Loss):
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
    def activation_function(logits):
        if logits.shape[1] == 3:
            return torch.softmax(logits, dim=1)
        else:
            return torch.sigmoid(logits)

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
