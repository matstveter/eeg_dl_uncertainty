import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from eegDlUncertainty.models.classifiers.swag import SWAG
from eegDlUncertainty.models.model_utils import mapping_avg_state_dict


class SWAClassifier(nn.Module):
    def __init__(self, pretrained_model, learning_rate, save_path, model_hyperparameters, name):
        super().__init__()

        self.model = copy.deepcopy(pretrained_model)
        self._learning_rate = learning_rate

        self._model_path = os.path.join(save_path, "model")
        self._model_hyperparameters = model_hyperparameters
        self._name = name

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)

    def fit(self, *, train_loader: DataLoader, val_loader: DataLoader, swa_epochs: int,
            device: torch.device, loss_fn: _Loss, swa_lr):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._learning_rate)
        # Create an averaged model for SWA
        swa_model = AveragedModel(self.model)
        self.to(device)

        # Set SWA start epoch and create SWA learning rate scheduler
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        for epoch in range(swa_epochs):
            print(
                f"\n-------------------------  SWA EPOCH {epoch + 1} / {swa_epochs}  -------------------------")

            self.train()
            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            swa_model.update_parameters(self.model)
            swa_scheduler.step()

        # Update batch normalization statistics for the SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        self.model.load_state_dict(mapping_avg_state_dict(averaged_model_state_dict=swa_model.state_dict()))
        path = os.path.join(self._model_path, f"swa_model")
        self.model.save_model(path=path)


class SWAGClassifier(nn.Module):

    def __init__(self, pretrained_model, learning_rate, save_path, model_hyperparameters, name):
        super().__init__()

        self.model = copy.deepcopy(pretrained_model)
        self._learning_rate = learning_rate

        self._model_path = os.path.join(save_path, "model")
        self._model_hyperparameters = model_hyperparameters
        self._name = name
        self.swag_model = SWAG(base=copy.deepcopy(pretrained_model), max_num_models=1, no_cov_mat=False)

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)

    def fit(self, *, train_loader: DataLoader, val_loader: DataLoader, swa_epochs: int,
            device: torch.device, loss_fn: _Loss, swa_lr):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._learning_rate)
        # Instead of a simple averaged model, create a SWAG model
        self.swag_model.to(device)
        self.model.to(device)

        # Set SWA start epoch and create SWA learning rate scheduler
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        for epoch in range(swa_epochs):
            print(
                f"\n-------------------------  SWA EPOCH {epoch + 1} / {swa_epochs}  -------------------------")

            self.model.train()
            for data, targets in train_loader:
                inputs, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            # Update SWAG model parameters after each epoch
            self.swag_model.collect_model(self.model)
            swa_scheduler.step()

        # After training, sample the weights and update batch normalization
        self.swag_model.sample(0.0)
        torch.optim.swa_utils.update_bn(train_loader, self.swag_model, device=device)

        # Save the state dictionary of the SWAG model
        self.model.load_state_dict(self.swag_model.state_dict())
        path = os.path.join(self._model_path, "swag_model")
        torch.save(self.model.state_dict(), path)

    def predict(self, test_loader, device, num_models=50):
        if self.swag_model is None:
            raise ValueError("SWAG model is None, call fit before trying to predict")

        predictions = []

        self.swag_model.to(device)
        self.swag_model.eval()

        with torch.no_grad():
            for _ in range(num_models):
                self.swag_model.sample(scale=0.5)

                for data, _ in test_loader:
                    inputs = data.to(device)
                    outputs = self.swag_model(inputs)
                    # todo Perhaps detach().cpu() ???
                    predictions.append(outputs)


        pass
